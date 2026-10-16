/*
 * HIP test for FLAT_ATOMIC_INC_X2 (ENC_FLAT, 64-bit atomic increment).
 *
 *   old = mem; mem = (old >= src) ? 0 : old + 1; return old (with sc0)
 *
 * Each thread targets its own 64-bit slot (no contention).  Every initial value
 * is kept well below the compare operand, so the result is unambiguously
 * old + 1 and the returned value is the original.  Both the returned old value
 * and the post-increment memory are checked.  Pattern C, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_atomic_inc_x2 flat_atomic_inc_x2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
static const uint64_t CMP=1000000000000ull;

__global__ void kernel(uint64_t*mem,const uint64_t*cmp,uint64_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(mem+i); uint64_t vcmp=cmp[i]; uint64_t old;
    __asm__ volatile("flat_atomic_inc_x2 %0, %1, %2 sc0\n\ts_waitcnt vmcnt(0)\n\t"
        :"=v"(old):"v"(addr),"v"(vcmp):"memory");
    old_out[i]=old;
}
int main(void){
    uint64_t*h_mem=(uint64_t*)malloc(N*8),*h_cmp=(uint64_t*)malloc(N*8),*h_old=(uint64_t*)malloc(N*8),*h_after=(uint64_t*)malloc(N*8);
    for(int i=0;i<N;i++){h_mem[i]=(uint64_t)(i*7+3);h_cmp[i]=CMP;}
    uint64_t*d_mem,*d_cmp,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*8));GPU_CHECK(hipMalloc(&d_cmp,N*8));GPU_CHECK(hipMalloc(&d_old,N*8));
    GPU_CHECK(hipMemcpy(d_mem,h_mem,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(d_cmp,h_cmp,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_cmp,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_old,d_old,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_after,d_mem,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint64_t eo=(uint64_t)(i*7+3),en=eo+1;
        if(h_old[i]!=eo||h_after[i]!=en){fprintf(stderr,"FAIL i=%d old %llu/%llu mem %llu/%llu\n",i,(unsigned long long)h_old[i],(unsigned long long)eo,(unsigned long long)h_after[i],(unsigned long long)en);pass=0;break;}}
    if(pass)printf("old[0]=%llu mem[0]=%llu\n",(unsigned long long)h_old[0],(unsigned long long)h_after[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
