/*
 * HIP test for FLAT_ATOMIC_ADD_F32 (ENC_FLAT, 32-bit float add atomic,
 * return-old).
 *
 *   old = MEM[addr]; MEM[addr] = old + DATA; VDST = old
 *
 * Each thread targets a disjoint slot (no contention).  Inputs are small
 * integer-valued floats so the sum is exact.  Pattern C, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_atomic_add_f32 flat_atomic_add_f32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(float*mem,const float*data,float*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(mem+i); float v=data[i],old;
    __asm__ volatile("flat_atomic_add_f32 %0, %1, %2 sc0\n\ts_waitcnt vmcnt(0) lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    old_out[i]=old;
}
int main(void){
    float*h_init=(float*)malloc(N*4),*h_data=(float*)malloc(N*4),*h_mem=(float*)malloc(N*4),*h_old=(float*)malloc(N*4);
    for(int i=0;i<N;i++){h_init[i]=(float)((i%50)-25);h_data[i]=(float)((i%37)-18);}
    float*d_mem,*d_data,*d_old;GPU_CHECK(hipMalloc(&d_mem,N*4));GPU_CHECK(hipMalloc(&d_data,N*4));GPU_CHECK(hipMalloc(&d_old,N*4));
    GPU_CHECK(hipMemcpy(d_mem,h_init,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(d_data,h_data,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_mem,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(h_old,d_old,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){float eo=h_init[i],em=h_init[i]+h_data[i];
        if(h_old[i]!=eo||h_mem[i]!=em){fprintf(stderr,"FAIL[%d] old=%f(exp %f) mem=%f(exp %f)\n",i,h_old[i],eo,h_mem[i],em);pass=0;break;}}
    if(pass)printf("old[0]=%f mem[0]=%f\n",h_old[0],h_mem[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
