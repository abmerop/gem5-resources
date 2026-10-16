/*
 * HIP test for S_SUB_U32 (ENC_SOP2, 32-bit unsigned subtract with borrow SCC).
 *   SDST = S0 - S1 ; SCC = (S1 > S0)   [borrow]
 * Pattern B (scalar). Matches gem5 sop2.cc exactly.
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_sub_u32 s_sub_u32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kernel(uint32_t s0,uint32_t s1,uint32_t*dout,uint32_t*sout){
    uint32_t d,scc;
    __asm__ volatile("s_sub_u32 %0, %2, %3\n\ts_cselect_b32 %1, 1, 0\n\t":"=s"(d),"=s"(scc):"s"(s0),"s"(s1));
    if(threadIdx.x==0){dout[blockIdx.x]=d;sout[blockIdx.x]=scc;}
}
int main(void){
    struct{uint32_t a,b;}cs[]={{100,40},{40,100},{0,0},{0xffffffffu,1},{1,0xffffffffu},{0x80000000u,0x80000000u},{12345,12345}};
    int nc=sizeof(cs)/sizeof(cs[0]);
    uint32_t*dd,*ds;GPU_CHECK(hipMalloc(&dd,nc*4));GPU_CHECK(hipMalloc(&ds,nc*4));
    for(int c=0;c<nc;c++){kernel<<<1,BLOCK_SIZE>>>(cs[c].a,cs[c].b,dd+c,ds+c);GPU_CHECK(hipGetLastError());}
    GPU_CHECK(hipDeviceSynchronize());
    uint32_t*hd=(uint32_t*)malloc(nc*4),*hs=(uint32_t*)malloc(nc*4);
    GPU_CHECK(hipMemcpy(hd,dd,nc*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hs,ds,nc*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<nc;c++){uint32_t ed=cs[c].a-cs[c].b,es=(cs[c].b>cs[c].a)?1:0;
        if(hd[c]!=ed||hs[c]!=es){fprintf(stderr,"FAIL %d d=%u(exp %u) scc=%u(exp %u)\n",c,hd[c],ed,hs[c],es);pass=0;break;}}
    if(pass)printf("d[0]=%u scc[0]=%u\n",hd[0],hs[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
