/* HIP test for GLOBAL_ATOMIC_ADD (32-bit VMEM atomic, return-old, gfx950).
 * Per-thread distinct 4-byte slot; host reference matches gem5 AtomicOp
 * functor. base only. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(uint32_t*mem,const uint32_t*data,uint32_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(&mem[i]);
    uint32_t v=data[i]; uint32_t old;
    __asm__ volatile("global_atomic_add %0, %1, %2, off sc0\n\ts_waitcnt vmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    old_out[i]=old;
}
int main(void){
    uint32_t*h_i=(uint32_t*)malloc(N*4),*h_d=(uint32_t*)malloc(N*4),*h_m=(uint32_t*)malloc(N*4),*h_o=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_i[i]=(uint32_t)(i*1000u+1);h_d[i]=(uint32_t)(i*7u+3);}
    uint32_t*d_mem,*d_data,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*4));GPU_CHECK(hipMalloc(&d_data,N*4));GPU_CHECK(hipMalloc(&d_old,N*4));
    GPU_CHECK(hipMemcpy(d_mem,h_i,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_d,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_m,d_mem,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_o,d_old,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint32_t a=h_i[i],b=h_d[i]; uint32_t em=(a + b),eo=a;
        if(h_o[i]!=eo||h_m[i]!=em){fprintf(stderr,"FAIL[%d] old=%u(e %u) mem=%u(e %u)\n",i,h_o[i],eo,h_m[i],em);pass=0;break;}}
    if(pass)printf("old[1]=%u mem[1]=%u\n",h_o[1],h_m[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
