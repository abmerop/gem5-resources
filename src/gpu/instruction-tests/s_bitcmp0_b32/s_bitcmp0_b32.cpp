/*
 * HIP test for S_BITCMP0_B32 (ENC_SOPC, scalar bit-is-zero test).
 *   SCC = (bit[S1 & 31] of S0 == 0). Pattern B (scalar). Matches gem5 sopc.cc.
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_bitcmp0_b32 s_bitcmp0_b32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kernel(uint32_t s0,uint32_t s1,uint32_t*out){
    uint32_t scc;
    __asm__ volatile("s_bitcmp0_b32 %1, %2\n\ts_cselect_b32 %0, 1, 0\n\t":"=s"(scc):"s"(s0),"s"(s1));
    if(threadIdx.x==0)out[blockIdx.x]=scc;
}
int main(void){
    struct{uint32_t s0,s1;}cs[]={{0x00000001u,0},{0x00000001u,1},{0xfffffffeu,0},{0x80000000u,31},{0x7fffffffu,31},{0x00010000u,16},{0xdeadbeefu,5},{0xdeadbeefu,6}};
    int nc=sizeof(cs)/sizeof(cs[0]);
    uint32_t*d_out;GPU_CHECK(hipMalloc(&d_out,nc*4));
    for(int c=0;c<nc;c++){kernel<<<1,BLOCK_SIZE>>>(cs[c].s0,cs[c].s1,d_out+c);GPU_CHECK(hipGetLastError());}
    GPU_CHECK(hipDeviceSynchronize());
    uint32_t*h_out=(uint32_t*)malloc(nc*4);GPU_CHECK(hipMemcpy(h_out,d_out,nc*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<nc;c++){uint32_t exp=(((cs[c].s0>>(cs[c].s1&31))&1u)==0)?1u:0u;
        if(h_out[c]!=exp){fprintf(stderr,"FAIL %d got %u exp %u\n",c,h_out[c],exp);pass=0;break;}}
    if(pass)printf("out[0]=%u out[1]=%u\n",h_out[0],h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
