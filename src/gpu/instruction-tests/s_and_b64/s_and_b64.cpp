/*
 * HIP test for S_AND_B64 (SOP2, 64-bit scalar bitwise AND).
 * gem5 (sop2.cc): SDST.u64 = S0.u64 & S1.u64; SCC = (SDST != 0).
 * Pattern B (scalar): operands are kernel-argument scalars (SGPRs).
 * Each block runs one wavefront; lane 0 writes the shared SDST result.
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
kern(uint64_t s0, uint64_t s1, uint64_t *out, uint32_t *scc_out)
{
    uint64_t sdst; uint32_t scc;
    __asm__ volatile(
        "s_and_b64 %0, %2, %3\n\t"
        "s_cselect_b32 %1, 1, 0\n\t"   /* capture SCC set by s_and_b64 */
        : "=s"(sdst), "=s"(scc)
        : "s"(s0), "s"(s1));
    if (threadIdx.x == 0) { out[blockIdx.x] = sdst; scc_out[blockIdx.x] = scc; }
}

int main(void)
{
    struct { uint64_t s0, s1; } cases[] = {
        { 0x0ULL,                0xFFFFFFFFFFFFFFFFULL },
        { 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL },
        { 0x00000000FFFFFFFFULL, 0xFFFFFFFF00000000ULL },
        { 0xDEADBEEFCAFEBABEULL, 0x0F0F0F0F0F0F0F0FULL },
        { 0x1234567890ABCDEFULL, 0xFEDCBA0987654321ULL },
        { 0x8000000000000000ULL, 0x8000000000000000ULL },
    };
    const int nc = (int)(sizeof(cases)/sizeof(cases[0]));
    uint64_t *d_out; uint32_t *d_scc;
    GPU_CHECK(hipMalloc(&d_out, nc*sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_scc, nc*sizeof(uint32_t)));
    for (int c = 0; c < nc; c++)
        kern<<<1,BLOCK_SIZE>>>(cases[c].s0, cases[c].s1, d_out+c, d_scc+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint64_t h_out[16]; uint32_t h_scc[16];
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*sizeof(uint64_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_scc,d_scc,nc*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass = 1;
    for (int c = 0; c < nc; c++) {
        uint64_t e = cases[c].s0 & cases[c].s1;
        uint32_t esc = e ? 1u : 0u;
        int ok = (h_out[c]==e) && (h_scc[c]==esc);
        printf("  0x%016llX & 0x%016llX = 0x%016llX (scc %u) %s\n",
               (unsigned long long)cases[c].s0,(unsigned long long)cases[c].s1,
               (unsigned long long)h_out[c], h_scc[c], ok?"OK":"FAIL");
        if (!ok) pass = 0;
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_out));GPU_CHECK(hipFree(d_scc));
    return pass ? 0 : 1;
}
