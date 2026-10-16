/*
 * HIP test for FLAT_LOAD_SHORT_D16 (ENC_FLAT): loads a 16-bit value into the
 * LOW half of VDST, preserving the high 16 bits.
 *   VDST = (VDST & 0xffff0000) | MEM[addr][15:0].
 * Sentinel preload verifies high-half preservation.  Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint16_t*data,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(&data[i]);
    uint32_t d=0xDEAD0000u; /* sentinel: high half must survive */
    __asm__ volatile("flat_load_short_d16 %0, %1\n\ts_waitcnt vmcnt(0) lgkmcnt(0)\n\t":"+v"(d):"v"(addr):"memory");
    out[i]=d;
}
int main(void){
    uint16_t*h=(uint16_t*)malloc(N*2); uint32_t*o=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h[i]=(uint16_t)(i*997+11);
    uint16_t*d; uint32_t*dO; GPU_CHECK(hipMalloc(&d,N*2));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(d,h,N*2,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(o,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1; for(int i=0;i<N;i++){uint32_t e=0xDEAD0000u|(uint32_t)h[i];
        if(o[i]!=e){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,o[i],e);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x\n",o[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
