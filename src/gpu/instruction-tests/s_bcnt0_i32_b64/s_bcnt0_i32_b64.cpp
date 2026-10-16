/*
 * HIP test for S_BCNT0_I32_B64 (SOP1, count zero bits of a 64-bit scalar).
 * gem5 (sop1.cc): SDST = 64 - popcount(S0.u64); SCC = (SDST != 0).
 * Pattern B (scalar): kernel-argument scalar operand (SGPR pair).
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
kern(uint64_t s0, int32_t *out, uint32_t *scc_out)
{
    int32_t sdst; uint32_t scc;
    __asm__ volatile(
        "s_bcnt0_i32_b64 %0, %2\n\t"
        "s_cselect_b32 %1, 1, 0\n\t"
        : "=s"(sdst), "=s"(scc) : "s"(s0));
    if (threadIdx.x == 0) { out[blockIdx.x] = sdst; scc_out[blockIdx.x] = scc; }
}

static int popcnt64(uint64_t v){ int c=0; while(v){c+=(int)(v&1);v>>=1;} return c; }

int main(void)
{
    uint64_t inputs[] = {
        0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL,
        0x00000000FFFFFFFFULL, 0xAAAAAAAAAAAAAAAAULL,
        0x0000000000000001ULL, 0x8000000000000000ULL,
        0xDEADBEEFCAFEBABEULL, 0x0102040810204080ULL,
    };
    const int nc = (int)(sizeof(inputs)/sizeof(inputs[0]));
    int32_t *d_out; uint32_t *d_scc;
    GPU_CHECK(hipMalloc(&d_out, nc*sizeof(int32_t)));
    GPU_CHECK(hipMalloc(&d_scc, nc*sizeof(uint32_t)));
    for (int c = 0; c < nc; c++)
        kern<<<1,BLOCK_SIZE>>>(inputs[c], d_out+c, d_scc+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    int32_t h_out[16]; uint32_t h_scc[16];
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*sizeof(int32_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_scc,d_scc,nc*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass = 1;
    for (int c = 0; c < nc; c++) {
        int32_t e = 64 - popcnt64(inputs[c]);
        uint32_t esc = e ? 1u : 0u;
        int ok = (h_out[c]==e) && (h_scc[c]==esc);
        printf("  0x%016llX -> %d zeros (scc %u) exp %d %s\n",
               (unsigned long long)inputs[c], h_out[c], h_scc[c], e, ok?"OK":"FAIL");
        if (!ok) pass = 0;
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_out));GPU_CHECK(hipFree(d_scc));
    return pass ? 0 : 1;
}
