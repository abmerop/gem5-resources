/*
 * HIP test for S_BITCMP1_B64 (SOPC: SCC = SSRC0[SSRC1 & 63]).
 * gem5 (sopc.cc): scc = bit(src0, src1&0x3f).  Pattern B, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint64_t v,uint32_t bit,uint32_t*out){
    uint32_t sdst;
    __asm__ volatile("s_bitcmp1_b64 %1, %2\n\ts_cselect_b32 %0, 1, 0\n\t":"=s"(sdst):"s"(v),"s"(bit):"scc");
    if(threadIdx.x==0) out[blockIdx.x]=sdst;
}
int main(void){
    uint64_t V[]={0x8000000000000001ull,0x8000000000000001ull,0x0ull,0xFFFFFFFFFFFFFFFFull,0x0000000100000000ull,0x0000000100000000ull};
    uint32_t B[]={0,63,5,40,32,31};
    const int NC=sizeof(B)/sizeof(B[0]);
    uint32_t*d; GPU_CHECK(hipMalloc(&d,NC*4));
    for(int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(V[c],B[c],d+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++){uint32_t exp=(uint32_t)((V[c]>>(B[c]&63))&1);
        if(h[c]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",c,h[c],exp);pass=0;}}
    if(pass)printf("out[0]=%u out[2]=%u\n",h[0],h[2]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
