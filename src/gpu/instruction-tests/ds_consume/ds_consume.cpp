/*
 * HIP test for DS_CONSUME (DS encoding, LDS consume-counter deallocation).
 *
 * gem5 (src/arch/amdgpu/common/insts/ds.cc): reads the LDS counter at M0.base,
 * atomically SUBTRACTS the number of active lanes, and broadcasts the
 * pre-operation value to VDST of every active lane (mirror of DS_APPEND).
 *
 * Self-check: launch one wavefront (64 active lanes) with the LDS counter
 * seeded to 128.
 *   Pass 1: every lane must receive 128 (pre-op value).
 *   Pass 2: every lane must receive 64  (counter decremented by 64 lanes).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int WAVE_SIZE=64;

__global__ void kernel(uint32_t*out1,uint32_t*out2){
    int lane=threadIdx.x;
    __shared__ uint32_t counter;
    if(lane==0) counter=128u;
    __syncthreads();
    uint32_t lds_addr;
    __asm__ volatile("v_readfirstlane_b32 %0, %1\n\t":"=s"(lds_addr):"v"((uint32_t)(uintptr_t)&counter));
    uint32_t r1,r2;
    __asm__ volatile(
        "s_mov_b32 m0, %2\n\t"
        "ds_consume %0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "ds_consume %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"=v"(r1),"=v"(r2):"s"(lds_addr):"m0","memory");
    out1[lane]=r1; out2[lane]=r2;
}
int main(void){
    uint32_t*h1=(uint32_t*)malloc(WAVE_SIZE*4),*h2=(uint32_t*)malloc(WAVE_SIZE*4);
    uint32_t*d1,*d2;
    GPU_CHECK(hipMalloc(&d1,WAVE_SIZE*4));GPU_CHECK(hipMalloc(&d2,WAVE_SIZE*4));
    kernel<<<1,WAVE_SIZE>>>(d1,d2);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h1,d1,WAVE_SIZE*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h2,d2,WAVE_SIZE*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<WAVE_SIZE;i++){
        if(h1[i]!=128u){fprintf(stderr,"FAIL pass1 lane %d got %u exp 128\n",i,h1[i]);pass=0;break;}
        if(h2[i]!=64u){fprintf(stderr,"FAIL pass2 lane %d got %u exp 64\n",i,h2[i]);pass=0;break;}}
    if(pass)printf("consume1=%u consume2=%u\n",h1[0],h2[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
