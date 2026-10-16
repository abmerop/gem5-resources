/*
 * HIP test for S_CMP_LG_I32 (ENC_SOPC, scalar not-equal compare).
 * gem5 (sopc.cc): SCC = (SSRC0 != SSRC1).
 * SCC captured via a following S_CSELECT_B32.
 * Pattern B (scalar): "s" operands force the SOPC encoding.
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
kern(int32_t s0, int32_t s1, uint32_t *out)
{
    uint32_t sdst;
    __asm__ volatile("s_cmp_lg_i32 %1, %2\n\t"
                     "s_cselect_b32 %0, 1, 0\n\t"
                     : "=s"(sdst) : "s"(s0), "s"(s1));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    struct { int32_t s0,s1; } cases[]={
        {5,5},{5,3},{-1,-1},{-1,1},{0,0},{2147483647,2147483647},
        {-2147483648,2147483647},{100,100},{-50,-50},{7,-7},
    };
    const int NC=sizeof(cases)/sizeof(cases[0]);
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out,NC*4));
    for(int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c].s0,cases[c].s1,d_out+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++){
        uint32_t exp=(cases[c].s0!=cases[c].s1)?1u:0u;
        if(h[c]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",c,h[c],exp);pass=0;}
    }
    if(pass)printf("out[0]=%u out[1]=%u\n",h[0],h[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
