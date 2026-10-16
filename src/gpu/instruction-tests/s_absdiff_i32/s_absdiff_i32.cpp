/*
 * HIP test for S_ABSDIFF_I32 (SOP2, absolute difference of signed 32-bit).
 * gem5 (sop2.cc): SDST = std::abs(S0.i32 - S1.i32); SCC = (SDST != 0).
 * Inputs are chosen so the signed subtraction and std::abs do not overflow.
 * Pattern B (scalar): kernel-argument scalar operands (SGPRs).
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
kern(int32_t s0, int32_t s1, int32_t *out, uint32_t *scc_out)
{
    int32_t sdst; uint32_t scc;
    __asm__ volatile(
        "s_absdiff_i32 %0, %2, %3\n\t"
        "s_cselect_b32 %1, 1, 0\n\t"
        : "=s"(sdst), "=s"(scc) : "s"(s0), "s"(s1));
    if (threadIdx.x == 0) { out[blockIdx.x] = sdst; scc_out[blockIdx.x] = scc; }
}

int main(void)
{
    struct { int32_t s0, s1; } cases[] = {
        { 0, 0 }, { 5, 5 }, { 10, 3 }, { 3, 10 },
        { -7, 4 }, { 4, -7 }, { -100, -250 }, { 1000000, -1000000 },
        { 2147483647, 2147483000 }, { -5, -5 },
    };
    const int nc = (int)(sizeof(cases)/sizeof(cases[0]));
    int32_t *d_out; uint32_t *d_scc;
    GPU_CHECK(hipMalloc(&d_out, nc*sizeof(int32_t)));
    GPU_CHECK(hipMalloc(&d_scc, nc*sizeof(uint32_t)));
    for (int c = 0; c < nc; c++)
        kern<<<1,BLOCK_SIZE>>>(cases[c].s0, cases[c].s1, d_out+c, d_scc+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    int32_t h_out[16]; uint32_t h_scc[16];
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*sizeof(int32_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_scc,d_scc,nc*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass = 1;
    for (int c = 0; c < nc; c++) {
        int64_t diff = (int64_t)cases[c].s0 - (int64_t)cases[c].s1;
        int32_t e = (int32_t)(diff < 0 ? -diff : diff);
        uint32_t esc = e ? 1u : 0u;
        int ok = (h_out[c]==e) && (h_scc[c]==esc);
        printf("  |%d - %d| = %d (scc %u) exp %d %s\n",
               cases[c].s0, cases[c].s1, h_out[c], h_scc[c], e, ok?"OK":"FAIL");
        if (!ok) pass = 0;
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_out));GPU_CHECK(hipFree(d_scc));
    return pass ? 0 : 1;
}
