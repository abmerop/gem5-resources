/*
 * HIP test for BUFFER_ATOMIC_UMIN_X2 (64-bit buffer atomic, return-old, gfx950/CDNA4).
 * Per-thread distinct 8-byte slot; host reference matches gem5 functor.
 * Build: hipcc -O2 --offload-arch=gfx950 -o buffer_atomic_umin_x2 buffer_atomic_umin_x2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
__global__ void kernel(uint64_t*mem,const uint64_t*data,uint64_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)mem; u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu); srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu; srd.w=0x00020000u;
    uint32_t voff=(uint32_t)(i*8); uint64_t v=data[i];
    __asm__ volatile("buffer_atomic_umin_x2 %0, %1, %2, 0 offen sc0\n\ts_waitcnt vmcnt(0)\n\t":"+v"(v):"v"(voff),"s"(srd):"memory");
    old_out[i]=v;
}
int main(void){
    uint64_t*h_init=(uint64_t*)malloc(N*sizeof(uint64_t)),*h_data=(uint64_t*)malloc(N*sizeof(uint64_t));
    uint64_t*h_mem=(uint64_t*)malloc(N*sizeof(uint64_t)),*h_old=(uint64_t*)malloc(N*sizeof(uint64_t));
    for(int i=0;i<N;i++){h_init[i]=((uint64_t)(i*2654435761u)<<12 | (uint64_t)(i*7u+1));h_data[i]=((uint64_t)(i*40503u+7u)<<9 | (uint64_t)(i^0x5a));}
    uint64_t*d_mem,*d_data,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*sizeof(uint64_t)));GPU_CHECK(hipMalloc(&d_data,N*sizeof(uint64_t)));GPU_CHECK(hipMalloc(&d_old,N*sizeof(uint64_t)));
    GPU_CHECK(hipMemcpy(d_mem,h_init,N*sizeof(uint64_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_data,N*sizeof(uint64_t),hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_mem,N*sizeof(uint64_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,N*sizeof(uint64_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint64_t eo=h_init[i]; uint64_t em=(h_init[i] < h_data[i] ? h_init[i] : h_data[i]);
        if(h_old[i]!=eo||h_mem[i]!=em){
            fprintf(stderr,"FAIL[%d] old=%llu(exp %llu) mem=%llu(exp %llu)\n",i,(unsigned long long)h_old[i],(unsigned long long)eo,(unsigned long long)h_mem[i],(unsigned long long)em);
            pass=0;break;}
    }
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
