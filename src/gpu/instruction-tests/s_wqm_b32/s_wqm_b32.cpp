/*
 * HIP test for S_WQM_B32 (SOP1, whole-quad-mode of a 32-bit scalar).
 * gem5 (sop1.cc via wholeQuadMode): for each nibble (4 bits) of S0, if any
 * bit of the nibble is set, all 4 bits of that nibble are set in SDST;
 * SCC = (SDST != 0). Pattern B (scalar): kernel-argument scalar (SGPR).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int BLOCK_SIZE = 64;

__global__ void
kern(uint32_t s0, uint32_t *out, uint32_t *scc_out)
{
    uint32_t sdst, scc;
    __asm__ volatile(
        "s_wqm_b32 %0, %2\n\t"
        "s_cselect_b32 %1, 1, 0\n\t"
        : "=s"(sdst), "=s"(scc) : "s"(s0));
    if (threadIdx.x == 0) { out[blockIdx.x] = sdst; scc_out[blockIdx.x] = scc; }
}

static uint32_t wqm(uint32_t v)
{
    uint32_t r = 0, mask = 0xF;
    for (int i = 0; i < 8; i++) { if (v & mask) r |= mask; mask <<= 4; }
    return r;
}

int main(void)
{
    uint32_t inputs[] = {
        0x00000000u, 0xFFFFFFFFu, 0x00000001u, 0x00000010u,
        0x12345678u, 0x80000000u, 0x0000000Fu, 0xA0B0C0D0u,
    };
    const int nc = (int)(sizeof(inputs)/sizeof(inputs[0]));
    uint32_t *d_out,*d_scc;
    GPU_CHECK(hipMalloc(&d_out, nc*sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_scc, nc*sizeof(uint32_t)));
    for (int c = 0; c < nc; c++)
        kern<<<1,BLOCK_SIZE>>>(inputs[c], d_out+c, d_scc+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint32_t h_out[16], h_scc[16];
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*sizeof(uint32_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_scc,d_scc,nc*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass = 1;
    for (int c = 0; c < nc; c++) {
        uint32_t e = wqm(inputs[c]); uint32_t esc = e ? 1u : 0u;
        int ok = (h_out[c]==e) && (h_scc[c]==esc);
        printf("  0x%08X -> 0x%08X (scc %u) exp 0x%08X %s\n",
               inputs[c], h_out[c], h_scc[c], e, ok?"OK":"FAIL");
        if (!ok) pass = 0;
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_out));GPU_CHECK(hipFree(d_scc));
    return pass ? 0 : 1;
}
