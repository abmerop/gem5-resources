/*
 * HIP test for S_QUADMASK_B64 (SOP1, quad mask reduce of a 64-bit scalar).
 * gem5 (sop1.cc via quadMask): for each nibble (4 bits) of S0, set one bit
 * in SDST if any bit of that nibble is set; SCC = (SDST != 0). 64-bit input
 * yields a 16-bit mask in the low half of the 64-bit SDST.
 * Pattern B (scalar): kernel-argument scalar operand (SGPR).
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
kern(uint64_t s0, uint64_t *out, uint32_t *scc_out)
{
    uint64_t sdst; uint32_t scc;
    __asm__ volatile(
        "s_quadmask_b64 %0, %2\n\t"
        "s_cselect_b32 %1, 1, 0\n\t"
        : "=s"(sdst), "=s"(scc) : "s"(s0));
    if (threadIdx.x == 0) { out[blockIdx.x] = sdst; scc_out[blockIdx.x] = scc; }
}

static uint64_t
ref(uint64_t v)
{
    uint64_t q = 0, mask = 0xF, qbit = 1;
    while (mask) { if (v & mask) q |= qbit; mask <<= 4; qbit <<= 1; }
    return q;
}

int main(void)
{
    uint64_t inputs[] = {
        0x0000000000000000ULL, 0xFFFFFFFFFFFFFFFFULL,
        0x0000000000000001ULL, 0x1111111111111111ULL,
        0x8000000000000000ULL, 0x000000000000000FULL,
        0xDEADBEEFCAFEBABEULL, 0x0102040810204080ULL,
    };
    const int nc = (int)(sizeof(inputs)/sizeof(inputs[0]));
    uint64_t *d_out; uint32_t *d_scc;
    GPU_CHECK(hipMalloc(&d_out, nc*sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_scc, nc*sizeof(uint32_t)));
    for (int c = 0; c < nc; c++)
        kern<<<1,BLOCK_SIZE>>>(inputs[c], d_out+c, d_scc+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint64_t h_out[16]; uint32_t h_scc[16];
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*sizeof(uint64_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_scc,d_scc,nc*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass = 1;
    for (int c = 0; c < nc; c++) {
        uint64_t e = ref(inputs[c]);
        uint32_t esc = e ? 1u : 0u;
        int ok = (h_out[c]==e) && (h_scc[c]==esc);
        printf("  0x%016llX -> 0x%016llX (scc %u) exp 0x%016llX %s\n",
               (unsigned long long)inputs[c],(unsigned long long)h_out[c],
               h_scc[c],(unsigned long long)e, ok?"OK":"FAIL");
        if (!ok) pass = 0;
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_out));GPU_CHECK(hipFree(d_scc));
    return pass ? 0 : 1;
}
