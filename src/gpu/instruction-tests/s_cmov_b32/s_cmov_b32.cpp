/*
 * HIP test for S_CMOV_B32 (SOP1: if (SCC) SDST = SSRC0).
 * gem5 (sop1.cc): reads SCC; if nonzero, sdst = src.  SCC is produced by a
 * preceding s_cmp_eq_u32 in the same asm block (cond==1 -> SCC=1).  Pattern B.
 * Variant: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint32_t orig, uint32_t val, uint32_t cond, uint32_t *out){
    uint32_t sdst=orig;
    __asm__ volatile("s_cmp_eq_u32 %2, 1\n\t"
                     "s_cmov_b32 %0, %1\n\t"
                     : "+s"(sdst) : "s"(val), "s"(cond) : "scc");
    if (threadIdx.x==0) out[blockIdx.x]=sdst;
}
int main(void){
    struct{uint32_t orig,val,cond;} cs[]={
        {0xAAAAAAAAu,0x12345678u,1},{0xAAAAAAAAu,0x12345678u,0},
        {0u,0xFFFFFFFFu,1},{7u,9u,0},{100u,200u,1},{100u,200u,0},
    };
    const int NC=sizeof(cs)/sizeof(cs[0]);
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out,NC*4));
    for(int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cs[c].orig,cs[c].val,cs[c].cond,d_out+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++){uint32_t exp=cs[c].cond?cs[c].val:cs[c].orig;
        if(h[c]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",c,h[c],exp);pass=0;}}
    if(pass)printf("out[0]=%u out[1]=%u\n",h[0],h[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));return pass?0:1;
}
