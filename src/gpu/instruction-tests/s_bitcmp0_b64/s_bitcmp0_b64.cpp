/*
 * HIP test for S_BITCMP0_B64 (ENC_SOPC, test that a selected bit is zero).
 * gem5 (sopc.cc): SCC = (bit SSRC1[5:0] of SSRC0.u64 == 0).
 * SCC captured via a following S_CSELECT_B32.
 * Pattern B (scalar): "s" operands (u64 value, u32 bit index) force the encoding.
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
kern(uint64_t s0, uint32_t s1, uint32_t *out)
{
    uint32_t sdst;
    __asm__ volatile("s_bitcmp0_b64 %1, %2\n\t"
                     "s_cselect_b32 %0, 1, 0\n\t"
                     : "=s"(sdst) : "s"(s0), "s"(s1));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    struct { uint64_t s0; uint32_t bit; } cases[]={
        {0x0000000000000000ull, 0},  {0xffffffffffffffffull, 0},
        {0x0000000000000001ull, 0},  {0x0000000000000001ull, 1},
        {0x8000000000000000ull, 63}, {0x8000000000000000ull, 62},
        {0x0123456789abcdefull, 32}, {0x0123456789abcdefull, 33},
        {0xaaaaaaaaaaaaaaaaull, 10}, {0xaaaaaaaaaaaaaaaaull, 11},
        {0x00000000ffffffffull, 31}, {0x00000000ffffffffull, 32},
    };
    const int NC=sizeof(cases)/sizeof(cases[0]);
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out,NC*4));
    for(int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c].s0,cases[c].bit,d_out+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++){
        uint32_t bit=(uint32_t)((cases[c].s0 >> (cases[c].bit & 63)) & 1ull);
        uint32_t exp=bit?0u:1u;
        if(h[c]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",c,h[c],exp);pass=0;}
    }
    if(pass)printf("out[0]=%u out[1]=%u\n",h[0],h[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
