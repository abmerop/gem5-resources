/*
 * HIP test for S_BITREPLICATE_B64_B32 (SOP1, replicate each bit twice).
 * gem5 (sop1.cc): for each bit i (0..31) of S0, set result bits 2*i and
 * 2*i+1 to that bit; SDST.u64 = result. Does not touch SCC.
 * Pattern B (scalar): 32-bit source -> 64-bit dest (SGPR pair).
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
kern(uint32_t s0, uint64_t *out)
{
    uint64_t sdst;
    __asm__ volatile("s_bitreplicate_b64_b32 %0, %1\n\t"
                     : "=s"(sdst) : "s"(s0));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

static uint64_t ref(uint32_t v)
{
    uint64_t r = 0;
    for (int i = 0; i < 32; i++) {
        uint64_t b = (v >> i) & 1u;
        r |= b << (i * 2);
        r |= b << (i * 2 + 1);
    }
    return r;
}

int main(void)
{
    uint32_t inputs[] = {
        0x00000000u, 0xFFFFFFFFu, 0x00000001u, 0x80000000u,
        0x0000FFFFu, 0xAAAAAAAAu, 0x12345678u, 0xDEADBEEFu,
    };
    const int nc = (int)(sizeof(inputs)/sizeof(inputs[0]));
    uint64_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, nc*sizeof(uint64_t)));
    for (int c = 0; c < nc; c++) kern<<<1,BLOCK_SIZE>>>(inputs[c], d_out+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint64_t h_out[16];
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*sizeof(uint64_t),hipMemcpyDeviceToHost));
    int pass = 1;
    for (int c = 0; c < nc; c++) {
        uint64_t e = ref(inputs[c]);
        int ok = (h_out[c]==e);
        printf("  0x%08X -> 0x%016llX exp 0x%016llX %s\n",
               inputs[c], (unsigned long long)h_out[c],
               (unsigned long long)e, ok?"OK":"FAIL");
        if (!ok) pass = 0;
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
