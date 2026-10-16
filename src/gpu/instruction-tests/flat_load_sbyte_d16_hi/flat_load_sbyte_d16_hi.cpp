/*
 * HIP test for FLAT_LOAD_SBYTE_D16_HI (ENC_FLAT, load signed byte into the high
 * 16 bits of the destination, preserving the low 16 bits).
 * gem5 (flat.cc completeAcc): replaceBits(vdst,31,16, (int16)(int8)byte).
 * The dst is seeded with a per-run sentinel and read-modified ("+v") so both the
 * loaded high half and the preserved low half are verified.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
#define SENT 0x0000BEEFu
__global__ void kern(const int8_t*g,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(g+i);
    uint32_t d=SENT;
    __asm__ volatile("flat_load_sbyte_d16_hi %0, %1\n\ts_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
                     : "+v"(d) : "v"(addr) : "memory");
    out[i]=d;
}
int main(void){
    int8_t*h_g=(int8_t*)malloc(N); uint32_t*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_g[i]=(int8_t)(i*7-100);
    int8_t*d_g; uint32_t*d_out;
    GPU_CHECK(hipMalloc(&d_g,N)); GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_g,h_g,N,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_g,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint16_t hi=(uint16_t)(int16_t)(int8_t)h_g[i];
        uint32_t exp=(SENT & 0x0000ffffu) | ((uint32_t)hi<<16);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
