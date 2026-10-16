/*
 * HIP test for GLOBAL_LOAD_LDS_SBYTE (ENC_FLAT_GLBL, direct global->LDS DMA
 * load of a signed byte, sign-extended to a 32-bit LDS dword).
 *
 * This is an LDS-DMA load: each active lane reads one byte from its global
 * address and the hardware writes it (sign-extended to 32 bits) into LDS at
 *   lds_addr = m0[17:2]*4 + inst_offset, then per lane + lane*4
 * (see gem5 vega/insts/op_encodings.hh ldsComplete<1,8>).  There is no VGPR
 * destination; m0 supplies the LDS base offset and the per-lane stride is
 * implicit.
 *
 * The compiler only emits the *unsigned* byte LDS-DMA via
 * __builtin_amdgcn_global_load_lds, so we use inline asm to force the SBYTE
 * (sign-extending) form.  lds[] is the kernel's only shared allocation, so it
 * sits at LDS offset 0 and m0 = 0 selects lds[0] as the lane-0 destination;
 * lane L lands in lds[L].
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o global_load_lds_sbyte global_load_lds_sbyte.cpp
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
static const int BLOCK_SIZE = 64;       /* one wavefront per block */

__global__ void
kernel_global_load_lds_sbyte(const int8_t *g, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ int lds[BLOCK_SIZE];     /* only shared alloc -> LDS offset 0 */
    int t = threadIdx.x;

    uint64_t addr = (uint64_t)(g + i);  /* per-lane global source address */

    /*
     * Feed the *real* LDS byte offset of lds[] into m0 instead of assuming 0,
     * and pass &lds[0] into the asm so the compiler sees the LDS array escape.
     * Without this, LLVM proves lds[] is never written by any IR store (the DMA
     * is invisible to it), folds the lds[] reads to undef, dead-code-eliminates
     * "out[i] = lds[t]" entirely, and allocates 0 bytes of group segment.
     */
    uint32_t lds_off = (uint32_t)(uintptr_t)lds;

    __asm__ volatile(
        "s_mov_b32 m0, %1\n\t"
        "global_load_lds_sbyte %0, off\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        :
        : "v"(addr), "s"(lds_off)
        : "memory", "m0");

    __syncthreads();
    out[i] = lds[t];
}

int
main(void)
{
    int8_t *h_g   = (int8_t *)malloc(N * sizeof(int8_t));
    int    *h_out = (int    *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
        h_g[i] = (int8_t)(i * 7 - 100);   /* mix of negative & positive */

    int8_t *d_g;
    int    *d_out;
    GPU_CHECK(hipMalloc(&d_g,   N * sizeof(int8_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_g, h_g, N * sizeof(int8_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_global_load_lds_sbyte, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_g, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int expected = (int)h_g[i];   /* sign-extended byte -> i32 */
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %d expected %d\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=%d out[%d]=%d\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_g)); GPU_CHECK(hipFree(d_out));
    free(h_g); free(h_out);
    return pass ? 0 : 1;
}
