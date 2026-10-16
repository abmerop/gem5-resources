/*
 * HIP test for S_CMPK_LE_U32 (SOPK: SCC = (sdst <= imm)).
 * gem5 (sopk.cc): uint32_t compare against zext16(SIMM16)=100.
 * Captured via S_CSELECT_B32.  Pattern B (scalar).  Variant: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint32_t v, uint32_t *out){
    uint32_t sdst;
    __asm__ volatile("s_cmpk_le_u32 %1, 100\n\t"
                     "s_cselect_b32 %0, 1, 0\n\t"
                     : "=s"(sdst) : "s"(v) : "scc");
    if (threadIdx.x==0) out[blockIdx.x]=sdst;
}
int main(void){
    uint32_t cases[]={0u,99u,100u,101u,50u,1000u,65535u,65536u,4294967295u,100u,1u,200u};
    const int NC=sizeof(cases)/sizeof(cases[0]);
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out,NC*4));
    for(int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c],d_out+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++){uint32_t exp=(cases[c] <= (uint32_t)100)?1u:0u;
        if(h[c]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",c,h[c],exp);pass=0;}}
    if(pass)printf("out[0]=%u out[3]=%u\n",h[0],h[3]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));return pass?0:1;
}
