/*
 * HIP test for S_CMOVK_I32 (SOPK: if (SCC) SDST = sext16(SIMM16)).
 * gem5 (sopk.cc): if SCC nonzero, sdst = sext<16>(imm)=-7.  SCC set by a
 * preceding s_cmp_eq_u32 in the same asm block.  Pattern B, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(int32_t orig,uint32_t cond,int32_t*out){
    int32_t sdst=orig;
    __asm__ volatile("s_cmp_eq_u32 %2, 1\n\ts_cmovk_i32 %0, 0xfff9\n\t":"+s"(sdst):"s"(0),"s"(cond):"scc");
    if(threadIdx.x==0) out[blockIdx.x]=sdst;
}
int main(void){
    struct{int32_t orig;uint32_t cond;} cs[]={{100,1},{100,0},{-5,1},{-5,0},{999,1},{999,0}};
    const int NC=sizeof(cs)/sizeof(cs[0]);
    int32_t*d; GPU_CHECK(hipMalloc(&d,NC*4));
    for(int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cs[c].orig,cs[c].cond,d+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    int32_t h[64]; GPU_CHECK(hipMemcpy(h,d,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++){int32_t exp=cs[c].cond?-7:cs[c].orig;
        if(h[c]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",c,h[c],exp);pass=0;}}
    if(pass)printf("out[0]=%d out[1]=%d\n",h[0],h[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
