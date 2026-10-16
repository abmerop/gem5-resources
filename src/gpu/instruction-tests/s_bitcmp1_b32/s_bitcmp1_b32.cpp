/*
 * HIP test for S_BITCMP1_B32 (SOPC: test that a selected bit is one).
 * gem5 (sopc.cc): SCC = (bit SSRC1[4:0] of SSRC0.u32 == 1).  Captured via
 * S_CSELECT_B32.  Pattern B (scalar).  Variants: base.
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
    __asm__ volatile("s_bitcmp1_b32 %1, %2\n\t"
                     "s_cselect_b32 %0, 1, 0\n\t"
                     : "=s"(sdst) : "s"(s0), "s"(s1));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    struct { uint32_t v; uint32_t bit; } cases[] = {
        {0x00000000u,0}, {0xffffffffu,0}, {0x00000001u,0}, {0x00000001u,1},
        {0x80000000u,31}, {0x80000000u,30}, {0x12345678u,4}, {0x12345678u,5},
        {0xaaaaaaaau,1}, {0xaaaaaaaau,0}, {0x0000ffffu,15}, {0x0000ffffu,16},
    };
    const int NC = sizeof(cases)/sizeof(cases[0]);
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out, NC*4));
    for (int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c].v, cases[c].bit, d_out+c);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int c=0;c<NC;c++) {
        uint32_t bit = (cases[c].v >> (cases[c].bit & 31u)) & 1u;
        uint32_t exp = bit ? 1u : 0u;
        if (h[c]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",c,h[c],exp);pass=0;}
    }
    if (pass) printf("out[2]=%u out[3]=%u\n", h[2], h[3]);
    printf("%s\n", pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
