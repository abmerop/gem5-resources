/*
 * HIP test for FLAT_ATOMIC_SMAX_X2 (ENC_FLAT, 64-bit signed-max atomic,
 * return-old).
 *
 *   old = MEM[addr]; MEM[addr] = max(old, DATA) (signed 64-bit); VDST = old
 *
 * Each thread targets a disjoint 64-bit slot.  Pattern C, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_atomic_smax_x2 flat_atomic_smax_x2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(int64_t*mem,const int64_t*data,int64_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(mem+i); int64_t v=data[i]; int64_t old;
    __asm__ volatile("flat_atomic_smax_x2 %0, %1, %2 sc0\n\ts_waitcnt vmcnt(0) lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    old_out[i]=old;
}
int main(void){
    int64_t*h_init=(int64_t*)malloc(N*8),*h_data=(int64_t*)malloc(N*8),*h_mem=(int64_t*)malloc(N*8),*h_old=(int64_t*)malloc(N*8);
    for(int i=0;i<N;i++){h_init[i]=(int64_t)(i*131-9000)*100000003LL;h_data[i]=(int64_t)(-i*177+5000)*100000007LL;}
    int64_t*d_mem,*d_data,*d_old;GPU_CHECK(hipMalloc(&d_mem,N*8));GPU_CHECK(hipMalloc(&d_data,N*8));GPU_CHECK(hipMalloc(&d_old,N*8));
    GPU_CHECK(hipMemcpy(d_mem,h_init,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(d_data,h_data,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_mem,N*8,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(h_old,d_old,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){int64_t eo=h_init[i],em=h_init[i]>h_data[i]?h_init[i]:h_data[i];
        if(h_old[i]!=eo||h_mem[i]!=em){fprintf(stderr,"FAIL[%d] old=%lld(exp %lld) mem=%lld(exp %lld)\n",i,(long long)h_old[i],(long long)eo,(long long)h_mem[i],(long long)em);pass=0;break;}}
    if(pass)printf("old[0]=%lld mem[0]=%lld\n",(long long)h_old[0],(long long)h_mem[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
