/*
 * HIP test for FLAT_ATOMIC_DEC (ENC_FLAT, 32-bit decrement atomic, return-old).
 *
 *   old = MEM[addr]; MEM[addr] = (old==0 || old>DATA) ? DATA : old-1; VDST=old
 *
 * We keep 0 < init <= DATA (DATA large) so the result is always old-1 under
 * both the ISA wrap semantics and gem5's plain-decrement functor.  Each thread
 * targets a disjoint slot.  Pattern C, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_atomic_dec flat_atomic_dec.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(uint32_t*mem,uint32_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(mem+i); uint32_t data=0xffffffffu,old;
    __asm__ volatile("flat_atomic_dec %0, %1, %2 sc0\n\ts_waitcnt vmcnt(0) lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(data):"memory");
    old_out[i]=old;
}
int main(void){
    uint32_t*h_init=(uint32_t*)malloc(N*4),*h_mem=(uint32_t*)malloc(N*4),*h_old=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++)h_init[i]=(uint32_t)(i+1000);
    uint32_t*d_mem,*d_old;GPU_CHECK(hipMalloc(&d_mem,N*4));GPU_CHECK(hipMalloc(&d_old,N*4));
    GPU_CHECK(hipMemcpy(d_mem,h_init,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_mem,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t eo=h_init[i],em=h_init[i]-1;
        if(h_old[i]!=eo||h_mem[i]!=em){fprintf(stderr,"FAIL[%d] old=%u(exp %u) mem=%u(exp %u)\n",i,h_old[i],eo,h_mem[i],em);pass=0;break;}}
    if(pass)printf("old[0]=%u mem[0]=%u\n",h_old[0],h_mem[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
