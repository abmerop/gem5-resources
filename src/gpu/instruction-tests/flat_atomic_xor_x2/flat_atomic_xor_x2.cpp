/*
 * HIP test for FLAT_ATOMIC_XOR_X2 (ENC_FLAT, 64-bit atomic XOR, return-old).
 *
 * FLAT_ATOMIC_XOR_X2 computes atomically on a 64-bit flat memory location:
 *
 *   tmp        = MEM[addr]
 *   MEM[addr]  = tmp ^ DATA          (64-bit)
 *   VDST       = tmp                  (old value, when return is requested)
 *
 * On CDNA4 (gfx950) the return form is requested with the sc0 cache modifier
 * (the assembler rejects glc/sc1 and requires sc0 for the return variant).
 * The instruction is forced with inline assembly; address, data and result
 * are all 64-bit VGPR pairs.  s_waitcnt vmcnt(0) stalls until the atomic
 * completes so the returned old value is valid.
 *
 * Each thread targets a disjoint 64-bit slot, so there is no contention and
 * both the returned old value and the post-op memory contents are
 * deterministic.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o flat_atomic_xor_x2 flat_atomic_xor_x2.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./flat_atomic_xor_x2
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

/*
 * kernel_flat_atomic_xor_x2
 *
 * mem[i] ^= data[i] (64-bit) atomically; the returned old value is written to
 * old_out[i].
 */
__global__ void
kernel_flat_atomic_xor_x2(uint64_t *mem, const uint64_t *data,
                          uint64_t *old_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t addr = (uint64_t)(mem + i);
    uint64_t vdata = data[i];
    uint64_t old;

    __asm__ volatile(
        "flat_atomic_xor_x2 %0, %1, %2 sc0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(old)
        : "v"(addr), "v"(vdata)
        : "memory");

    old_out[i] = old;
}

int
main(void)
{
    uint64_t *h_init = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_data = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_mem  = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_old  = (uint64_t *)malloc(N * sizeof(uint64_t));

    for (int i = 0; i < N; i++) {
        h_init[i] = 0x0123456789abcdefULL ^ ((uint64_t)i * 0x100000001ULL);
        h_data[i] = 0xdeadbeefcafef00dULL + ((uint64_t)i << 32);
    }

    uint64_t *d_mem, *d_data, *d_old;
    GPU_CHECK(hipMalloc(&d_mem,  N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_data, N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_old,  N * sizeof(uint64_t)));

    GPU_CHECK(hipMemcpy(d_mem,  h_init, N * sizeof(uint64_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data, h_data, N * sizeof(uint64_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_atomic_xor_x2,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_mem, d_data, d_old, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_mem, d_mem, N * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old, d_old, N * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint64_t exp_old = h_init[i];
        uint64_t exp_mem = h_init[i] ^ h_data[i];
        if (h_old[i] != exp_old) {
            fprintf(stderr, "FAIL old[%d]: got 0x%016llx exp 0x%016llx\n",
                    i, (unsigned long long)h_old[i],
                    (unsigned long long)exp_old);
            pass = 0; break;
        }
        if (h_mem[i] != exp_mem) {
            fprintf(stderr, "FAIL mem[%d]: got 0x%016llx exp 0x%016llx\n",
                    i, (unsigned long long)h_mem[i],
                    (unsigned long long)exp_mem);
            pass = 0; break;
        }
    }

    if (pass) {
        printf("old[1]=0x%016llx mem[1]=0x%016llx\n",
               (unsigned long long)h_old[1], (unsigned long long)h_mem[1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_mem)); GPU_CHECK(hipFree(d_data));
    GPU_CHECK(hipFree(d_old));
    free(h_init); free(h_data); free(h_mem); free(h_old);

    return pass ? 0 : 1;
}
