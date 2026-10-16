/*
 * HIP test for S_CMOV_B64 (SOP1: if (SCC) SDST[63:0] = SSRC0[63:0]).
 * SCC set by a preceding s_cmp_eq_u32.  Pattern B, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint64_t orig,uint64_t val,uint32_t cond,uint64_t*out){
    uint64_t sdst=orig;
    __asm__ volatile("s_cmp_eq_u32 %2, 1\n\ts_cmov_b64 %0, %1\n\t":"+s"(sdst):"s"(val),"s"(cond):"scc");
    if(threadIdx.x==0) out[blockIdx.x]=sdst;
}
int main(void){
    struct{uint64_t orig,val;uint32_t cond;} cs[]={
        {0x1111222233334444ull,0xAAAABBBBCCCCDDDDull,1},
        {0x1111222233334444ull,0xAAAABBBBCCCCDDDDull,0},
        {0ull,0xFFFFFFFFFFFFFFFFull,1},{7ull,9ull,0}};
    const int NC=sizeof(cs)/sizeof(cs[0]);
    uint64_t*d; GPU_CHECK(hipMalloc(&d,NC*8));
    for(int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cs[c].orig,cs[c].val,cs[c].cond,d+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint64_t h[64]; GPU_CHECK(hipMemcpy(h,d,NC*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++){uint64_t exp=cs[c].cond?cs[c].val:cs[c].orig;
        if(h[c]!=exp){fprintf(stderr,"FAIL[%d] got %llu exp %llu\n",c,(unsigned long long)h[c],(unsigned long long)exp);pass=0;}}
    if(pass)printf("out[0]=%llu out[1]=%llu\n",(unsigned long long)h[0],(unsigned long long)h[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
