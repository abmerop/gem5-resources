/*
 * HIP test for S_SCRATCH_LOAD_DWORD (ENC_SMEM, scalar 1-dword scratch load).
 * Pattern B.
 *
 * gem5 (vega/insts/smem.hh + Inst_SMEM::calcScratchAddr): the final scalar
 * address is
 *   vaddr = SBASE + inst_offset + scaled_offset*64
 * i.e. S_SCRATCH_LOAD_DWORD reads the SGPR SDATA from the 64-bit address held
 * in the SBASE SGPR pair.  There is no implicit private base, so pointing
 * SBASE at an ordinary global allocation makes this a plain scalar load.
 *
 * To exercise the load against known-good data we first WRITE the slot with
 * S_SCRATCH_STORE_DWORD (also implemented in gem5) and then read it back with
 * S_SCRATCH_LOAD_DWORD, verifying the round-trip.  Each wavefront owns its own
 * 4-byte slot (SBASE = scratch + wf) so the wave-uniform ops never race.
 *
 * Assembler forms (validated with llvm-mc -mcpu=gfx950):
 *   s_scratch_store_dword s2, s[0:1], 0x0
 *   s_scratch_load_dword  s3, s[0:1], 0x0 glc
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_scratch_load_dword \
 *        s_scratch_load_dword.cpp
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

static const int NUM_WF = 64, BLOCK_SIZE = 64; /* one wavefront per block */

__global__ void
kernel_s_scratch_load_dword(uint32_t base_val, uint32_t *scratch,
                            uint32_t *out)
{
    int wf = blockIdx.x;
    uint32_t data = base_val + (uint32_t)wf;
    /* per-wavefront private 4-byte slot -> wave-uniform SBASE */
    uint64_t sbase = (uint64_t)(scratch + wf);
    uint32_t res;

    __asm__ volatile(
        "s_scratch_store_dword %1, %2, 0x0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "s_scratch_load_dword  %0, %2, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "=s"(res)
        : "s"(data), "s"(sbase)
        : "memory");

    if (threadIdx.x == 0)
        out[wf] = res;
}

int
main(void)
{
    const uint32_t base_val = 0xB0DE0000u;
    uint32_t *h_out = (uint32_t *)malloc(NUM_WF * sizeof(uint32_t));

    uint32_t *d_out, *d_scratch;
    GPU_CHECK(hipMalloc(&d_out, NUM_WF * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_scratch, NUM_WF * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_scratch, 0, NUM_WF * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_out, 0, NUM_WF * sizeof(uint32_t)));

    hipLaunchKernelGGL(kernel_s_scratch_load_dword, dim3(NUM_WF),
                       dim3(BLOCK_SIZE), 0, 0, base_val, d_scratch, d_out);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, NUM_WF * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int wf = 0; wf < NUM_WF; wf++) {
        uint32_t exp = base_val + (uint32_t)wf;
        if (h_out[wf] != exp) {
            fprintf(stderr, "FAIL wf%d: got 0x%08x expected 0x%08x\n", wf,
                    h_out[wf], exp);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], NUM_WF - 1,
               h_out[NUM_WF - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    GPU_CHECK(hipFree(d_scratch));
    free(h_out);
    return pass ? 0 : 1;
}
