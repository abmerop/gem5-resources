/* HIP test for S_GETREG_B32 (SOPK, read a hardware register field into an SGPR).
 *
 * gem5 (Inst_SOPK__S_GETREG_B32::execute in common/insts/sopk.cc):
 *   hwregId = SIMM16 & 0x3f; offset = (SIMM16>>6)&31; size = ((SIMM16>>11)&31)+1
 *   sdst = (getHwReg(hwregId) & (((1<<size)-1)<<offset)) >> offset
 *
 * We read HW_REG_SH_MEM_BASES (id 0xf).  gem5's shader initializes this
 * register deterministically for gfx950:
 *   setHwReg(HW_REG_SH_MEM_BASES, 0x00010002)   (src/gpu-compute/shader.cc)
 * This is the only hwReg gem5 pre-populates (getHwReg panics on unset ids),
 * which makes it the reliable, deterministic choice for a self-verifying test.
 *
 * We read the two 16-bit halves separately: offset 0 size 16 -> 0x0002 (low),
 * offset 16 size 16 -> 0x0001 (high).  (A full offset-0 size-32 read is avoided
 * because gem5's mask "(1U << size) - 1" is undefined for size==32 and returns
 * a zero mask on the x86 host.)
 *
 * Pattern B (scalar / wavefront-uniform): one wavefront per block; only lane 0
 * writes the wavefront-wide result.  base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_getreg_b32 s_getreg_b32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NUM_WF = 64;

__global__ void
kernel(uint32_t *out)
{
    int wf = blockIdx.x;
    uint32_t lo, hi;
    __asm__ volatile(
        "s_getreg_b32 %0, hwreg(HW_REG_SH_MEM_BASES, 0, 16)\n\t"
        : "=s"(lo));
    __asm__ volatile(
        "s_getreg_b32 %0, hwreg(HW_REG_SH_MEM_BASES, 16, 16)\n\t"
        : "=s"(hi));
    if (threadIdx.x == 0)
        out[wf] = (hi << 16) | (lo & 0xffffu);
}

int
main(void)
{
    const uint32_t EXPECTED = 0x00010002u;
    uint32_t h_out[NUM_WF];
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, NUM_WF * sizeof(uint32_t)));

    kernel<<<NUM_WF, 64>>>(d_out);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, NUM_WF * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int w = 0; w < NUM_WF; w++) {
        if (h_out[w] != EXPECTED) {
            fprintf(stderr, "FAIL[%d] got 0x%08x expected 0x%08x\n",
                    w, h_out[w], EXPECTED);
            pass = 0;
            break;
        }
    }
    if (pass)
        printf("out[0]=0x%08x\n", h_out[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
