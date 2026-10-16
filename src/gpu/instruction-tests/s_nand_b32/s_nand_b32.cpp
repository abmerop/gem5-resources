/*
 * HIP test for S_NAND_B32 (ENC_SOP2, scalar bitwise NAND).
 * gem5 (sop2.cc): SDST = ~(SSRC0 & SSRC1); SCC = (SDST != 0).
 * Pattern B (scalar): uniform kernel args in SGPRs; "=s"/"s" force the encoding.
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
    __asm__ volatile("s_nand_b32 %0, %1, %2\n\t" : "=s"(sdst) : "s"(s0), "s"(s1));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    struct { uint32_t s0,s1; } cases[]={
        {0,0},{0xffffffffu,0xffffffffu},{0xffffffffu,0},{0x12345678u,0x0f0f0f0fu},
        {0xdeadbeefu,0xcafebabeu},{0xaaaaaaaau,0x55555555u},{0x80000000u,0x80000000u},
        {0x00ff00ffu,0xff00ff00u},
    };
    const int NC=sizeof(cases)/sizeof(cases[0]);
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out,NC*4));
    for(int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c].s0,cases[c].s1,d_out+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++){
        uint32_t exp=~(cases[c].s0 & cases[c].s1);
        if(h[c]!=exp){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",c,h[c],exp);pass=0;}
    }
    if(pass)printf("out[3]=%08x\n",h[3]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
