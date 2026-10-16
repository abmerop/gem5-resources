/*
 * HIP test for S_SCRATCH_LOAD_DWORDX4 (ENC_SMEM, scalar 4-dword scratch
 * load).  Pattern B.
 *
 * gem5 (smem.hh + Inst_SMEM::calcScratchAddr): the final scalar address is
 *   vaddr = SBASE + inst_offset + scaled_offset*64
 * i.e. S_SCRATCH_LOAD_DWORDX4 reads the 16-byte SGPR quad SDATA from the
 * 64-bit address held in the SBASE SGPR pair.  There is no implicit private
 * base, so pointing SBASE at an ordinary global allocation makes this a plain
 * scalar 4-dword load from that address.  This is a READ-only test (the scalar
 * scratch STORE write path is a known gem5 protocol gap); the source quad is
 * initialized from the host so no scalar store is needed.
 *
 * Each wavefront gets its own 16-byte slot (SBASE = src + wf*4 dwords) so the
 * wave-uniform load is well defined, then loads {d0,d1,d2,d3} and thread 0
 * writes them to `out`.  The host verifies out == src exactly.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   s_scratch_load_dwordx4 s[4:7], s[0:1], 0x0 glc
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_scratch_load_dwordx4 \
 *        s_scratch_load_dwordx4.cpp
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
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

static const int NUM_WF = 64, BLOCK_SIZE = 64; /* one wavefront per block */

__global__ void
kernel_s_scratch_load_dwordx4(const uint32_t *src, uint32_t *out)
{
    int wf = blockIdx.x;
    /* per-wavefront private 16-byte slot -> wave-uniform SBASE */
    uint64_t sbase = (uint64_t)(src + wf * 4);
    u32x4 res;

    __asm__ volatile(
        "s_scratch_load_dwordx4 %0, %1, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "=s"(res)
        : "s"(sbase)
        : "memory");

    if (threadIdx.x == 0) {
        out[wf * 4 + 0] = res.x;
        out[wf * 4 + 1] = res.y;
        out[wf * 4 + 2] = res.z;
        out[wf * 4 + 3] = res.w;
    }
}

int
main(void)
{
    const int total = NUM_WF * 4;
    uint32_t *h_src = (uint32_t *)malloc(total * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(total * sizeof(uint32_t));

    for (int i = 0; i < total; i++)
        h_src[i] = 0xC0DE0000u + (uint32_t)i * 7u;

    uint32_t *d_src, *d_out;
    GPU_CHECK(hipMalloc(&d_src, total * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, total * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_src, h_src, total * sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_out, 0, total * sizeof(uint32_t)));

    hipLaunchKernelGGL(kernel_s_scratch_load_dwordx4, dim3(NUM_WF),
                       dim3(BLOCK_SIZE), 0, 0, d_src, d_out);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, total * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < total; i++) {
        if (h_out[i] != h_src[i]) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n", i,
                    h_out[i], h_src[i]);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], total - 1,
               h_out[total - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src));
    GPU_CHECK(hipFree(d_out));
    free(h_src);
    free(h_out);
    return pass ? 0 : 1;
}
