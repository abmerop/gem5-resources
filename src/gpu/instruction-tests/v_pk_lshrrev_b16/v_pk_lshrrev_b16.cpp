/*
 * HIP test for V_PK_LSHRREV_B16 (ENC_VOP3P, packed 16-bit logical
 * shift-right-reverse).
 *
 * V_PK_LSHRREV_B16 operates independently on the two 16-bit halves of its
 * packed operands (per gem5 src/arch/amdgpu/common/insts/vop3p.cc):
 *
 *   for h in {lo, hi}:
 *     DST[h] = SRC1[h] >> (SRC0[h] & 0xf)
 *
 * i.e. "reverse" operand order — SRC0 supplies the shift amount and SRC1 the
 * value being shifted.  The instruction is forced with inline assembly; the
 * two packed 16-bit lanes live in the low/high halves of 32-bit VGPRs.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_pk_lshrrev_b16 v_pk_lshrrev_b16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_pk_lshrrev_b16
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
static const int BLOCK_SIZE = 64;

/* packed: src0 = shift amounts, src1 = values */
__global__ void
kernel_v_pk_lshrrev_b16(const uint32_t *shift, const uint32_t *val,
                        uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t vs0 = shift[i];
    uint32_t vs1 = val[i];
    uint32_t vd;
    __asm__ volatile(
        "v_pk_lshrrev_b16 %0, %1, %2\n\t"
        : "=v"(vd)
        : "v"(vs0), "v"(vs1));

    out[i] = vd;
}

static uint32_t
pack(uint16_t lo, uint16_t hi)
{
    return ((uint32_t)hi << 16) | (uint32_t)lo;
}

int
main(void)
{
    uint32_t *h_shift = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_val   = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out   = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        uint16_t sl = (uint16_t)(i & 0xf);
        uint16_t sh = (uint16_t)((i >> 1) & 0xf);
        uint16_t vl = (uint16_t)(0xC000 + i);
        uint16_t vh = (uint16_t)(0x8001 + i * 7);
        h_shift[i] = pack(sl, sh);
        h_val[i]   = pack(vl, vh);
    }

    uint32_t *d_shift, *d_val, *d_out;
    GPU_CHECK(hipMalloc(&d_shift, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_val,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out,   N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_shift, h_shift, N * sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_val, h_val, N * sizeof(uint32_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_pk_lshrrev_b16,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_shift, d_val, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint16_t sl = (uint16_t)(h_shift[i] & 0xffff);
        uint16_t sh = (uint16_t)(h_shift[i] >> 16);
        uint16_t vl = (uint16_t)(h_val[i] & 0xffff);
        uint16_t vh = (uint16_t)(h_val[i] >> 16);
        uint16_t rl = vl >> (sl & 0xf);
        uint16_t rh = vh >> (sh & 0xf);
        uint32_t expected = pack(rl, rh);
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], expected);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=0x%08x out[%d]=0x%08x\n",
               h_out[0], N - 1, h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_shift)); GPU_CHECK(hipFree(d_val));
    GPU_CHECK(hipFree(d_out));
    free(h_shift); free(h_val); free(h_out);

    return pass ? 0 : 1;
}
