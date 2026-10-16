/* HIP test for FLAT_ATOMIC_DEC_X2 (FLAT 64-bit atomic dec, return-old, gfx950).
 * Per-thread distinct 8-byte slot. AtomicOpDecWraparound: keep 0<init<=cmp so
 * mem=init-1, old=init. base only. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(uint64_t*mem,uint64_t cmp,uint64_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(&mem[i]);
    uint64_t v=cmp,old;
    __asm__ volatile("flat_atomic_dec_x2 %0, %1, %2 sc0\n\ts_waitcnt vmcnt(0) lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    old_out[i]=old;
}
int main(void){
    uint64_t*h_init=(uint64_t*)malloc(N*8),*h_mem=(uint64_t*)malloc(N*8),*h_old=(uint64_t*)malloc(N*8);
    const uint64_t CMP=0xFFFFFFFFFFULL;
    for(int i=0;i<N;i++) h_init[i]=(uint64_t)(i+1)*1000003ULL;
    uint64_t*d_mem,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*8));GPU_CHECK(hipMalloc(&d_old,N*8));
    GPU_CHECK(hipMemcpy(d_mem,h_init,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,CMP,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_mem,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint64_t eo=h_init[i],em=h_init[i]-1;
        if(h_old[i]!=eo||h_mem[i]!=em){fprintf(stderr,"FAIL[%d] old=%llu(e %llu) mem=%llu(e %llu)\n",i,(unsigned long long)h_old[i],(unsigned long long)eo,(unsigned long long)h_mem[i],(unsigned long long)em);pass=0;break;}}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
