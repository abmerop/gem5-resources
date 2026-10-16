/*
 * HIP test for S_PACK_HH_B32_B16 (SOP2, scalar half-word pack High/High).
 * gem5 (sop2.cc): SDST = ((S0 >> 16) & 0x0000FFFF) | (S1 & 0xFFFF0000).
 * i.e. high 16 bits of S0 -> low half; high 16 bits of S1 -> high half.
 * Does not touch SCC. Pattern B (scalar): kernel-argument scalars (SGPRs).
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
kern(uint32_t s0, uint32_t s1, uint32_t *out)
{
    uint32_t sdst;
    __asm__ volatile("s_pack_hh_b32_b16 %0, %1, %2\n\t"
                     : "=s"(sdst) : "s"(s0), "s"(s1));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    struct { uint32_t s0, s1; } cases[] = {
        { 0x00000000u, 0x00000000u },
        { 0xABCD0000u, 0x12340000u },
        { 0xFFFFFFFFu, 0x00000000u },
        { 0x00000000u, 0xFFFFFFFFu },
        { 0xDEADBEEFu, 0xCAFEBABEu },
        { 0x12345678u, 0x9ABCDEF0u },
    };
    const int nc = (int)(sizeof(cases)/sizeof(cases[0]));
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, nc*sizeof(uint32_t)));
    for (int c = 0; c < nc; c++)
        kern<<<1,BLOCK_SIZE>>>(cases[c].s0, cases[c].s1, d_out+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint32_t h_out[16];
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass = 1;
    for (int c = 0; c < nc; c++) {
        uint32_t e = ((cases[c].s0 >> 16) & 0x0000FFFFu) | (cases[c].s1 & 0xFFFF0000u);
        int ok = (h_out[c]==e);
        printf("  s0=0x%08X s1=0x%08X -> 0x%08X exp 0x%08X %s\n",
               cases[c].s0, cases[c].s1, h_out[c], e, ok?"OK":"FAIL");
        if (!ok) pass = 0;
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
