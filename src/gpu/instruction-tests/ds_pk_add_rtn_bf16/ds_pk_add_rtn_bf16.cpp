/*
 * HIP test for DS_PK_ADD_RTN_BF16 (ENC_DS, packed bf16 LDS atomic add,
 * return-old).
 *
 * DS_PK_ADD_RTN_BF16 computes, atomically on a 32-bit LDS dword that holds
 * two packed bf16 values:
 *
 *   tmp           = LDS[addr]                 (two packed bf16 lanes)
 *   LDS[addr].lo  = tmp.lo + DATA.lo          (bf16 add)
 *   LDS[addr].hi  = tmp.hi + DATA.hi          (bf16 add)
 *   VDST          = tmp                        (the OLD packed value)
 *
 * There is no HIP intrinsic / atomicAdd overload for packed bf16, so the
 * instruction is forced with inline assembly.  The DS address operand is the
 * 32-bit LDS byte offset; we obtain it by casting the __shared__ pointer to a
 * local (address-space 3) pointer, whose representation on AMDGCN is exactly
 * that 32-bit offset.
 *
 * Each thread owns a distinct LDS slot, so there is no real contention and the
 * returned old value / final slot value are deterministic.
 *
 * Inputs are exactly-representable small-integer bf16 values whose sums are
 * also exactly representable, so bf16 addition is bit-exact and verification
 * can use integer comparison on the packed bits.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o ds_pk_add_rtn_bf16 ds_pk_add_rtn_bf16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./ds_pk_add_rtn_bf16
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                   \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N          = 256;
static const int BLOCK_SIZE = 64;       /* one wavefront per block */

/*
 * kernel_ds_pk_add_rtn_bf16
 *
 * Each thread seeds its own LDS slot with init[i], then atomically adds
 * data[i] (packed bf16) into it, capturing the returned old value.  Both the
 * returned old value and the post-add slot contents are written to global
 * memory for host verification.
 */
__global__ void
kernel_ds_pk_add_rtn_bf16(const uint32_t *init, const uint32_t *data,
                          uint32_t *old_out, uint32_t *new_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ uint32_t lds[BLOCK_SIZE];
    int t = threadIdx.x;

    lds[t] = init[i];
    __syncthreads();

    /* 32-bit LDS byte offset of lds[t]: cast generic -> local pointer. */
    __attribute__((address_space(3))) uint32_t *lp =
        (__attribute__((address_space(3))) uint32_t *)(&lds[t]);
    uint32_t addr = (uint32_t)(uintptr_t)lp;

    uint32_t vdata = data[i];
    uint32_t old;
    __asm__ volatile(
        "ds_pk_add_rtn_bf16 %0, %1, %2\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "=v"(old)
        : "v"(addr), "v"(vdata)
        : "memory");

    __syncthreads();
    old_out[i] = old;
    new_out[i] = lds[t];
}

/* Truncate f32 -> bf16 (exact for the small integers used here). */
static uint16_t
to_bf16(float f)
{
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return (uint16_t)(u >> 16);
}

static uint32_t
pack_bf16(float lo, float hi)
{
    return ((uint32_t)to_bf16(hi) << 16) | (uint32_t)to_bf16(lo);
}

int
main(void)
{
    uint32_t *h_init = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_data = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_old  = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_new  = (uint32_t *)malloc(N * sizeof(uint32_t));

    /* small exact integers; lane sums stay <= ~32 (exact in bf16). */
    float lo0[N], hi0[N], lod[N], hid[N];
    for (int i = 0; i < N; i++) {
        lo0[i] = (float)(i % 8);
        hi0[i] = (float)(i % 5 + 1);
        lod[i] = (float)(i % 3 + 1);
        hid[i] = (float)(i % 7);
        h_init[i] = pack_bf16(lo0[i], hi0[i]);
        h_data[i] = pack_bf16(lod[i], hid[i]);
    }

    uint32_t *d_init, *d_data, *d_old, *d_new;
    GPU_CHECK(hipMalloc(&d_init, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_data, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_old,  N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_new,  N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_init, h_init, N * sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data, h_data, N * sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_ds_pk_add_rtn_bf16,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_init, d_data, d_old, d_new, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_old, d_old, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_new, d_new, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t exp_old = h_init[i];
        uint32_t exp_new = pack_bf16(lo0[i] + lod[i], hi0[i] + hid[i]);
        if (h_old[i] != exp_old) {
            fprintf(stderr, "FAIL old[%d]: got 0x%08x exp 0x%08x\n",
                    i, h_old[i], exp_old);
            pass = 0; break;
        }
        if (h_new[i] != exp_new) {
            fprintf(stderr, "FAIL new[%d]: got 0x%08x exp 0x%08x\n",
                    i, h_new[i], exp_new);
            pass = 0; break;
        }
    }

    if (pass) {
        printf("old[1]=0x%08x new[1]=0x%08x\n", h_old[1], h_new[1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_init)); GPU_CHECK(hipFree(d_data));
    GPU_CHECK(hipFree(d_old));  GPU_CHECK(hipFree(d_new));
    free(h_init); free(h_data); free(h_old); free(h_new);

    return pass ? 0 : 1;
}
