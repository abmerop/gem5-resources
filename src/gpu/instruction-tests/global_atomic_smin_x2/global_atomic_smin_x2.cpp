/*
 * HIP test for GLOBAL_ATOMIC_SMIN_X2 (ENC_FLAT_GLBL, 64-bit signed-min atomic,
 * return-old).  Pattern C.
 *
 *   tmp        = MEM[addr]
 *   MEM[addr]  = smin(tmp, DATA)       (64-bit signed min)
 *   VDST       = tmp                   (old value, with sc0 return)
 *
 * Operand form (llvm-mc -mcpu=gfx950):
 *   global_atomic_smin_x2 vdst, vaddr, vdata, off sc0
 * Each thread targets its own 8-byte slot so old/new are deterministic.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o global_atomic_smin_x2 \
 *        global_atomic_smin_x2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(int64_t*mem,const int64_t*data,int64_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(&mem[i]);
    int64_t vsrc=data[i],old;
    __asm__ volatile(
        "global_atomic_smin_x2 %0, %1, %2, off sc0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :"=v"(old):"v"(addr),"v"(vsrc):"memory");
    old_out[i]=old;
}
int main(void){
    int64_t*h_init=(int64_t*)malloc(N*8),*h_data=(int64_t*)malloc(N*8);
    int64_t*h_mem=(int64_t*)malloc(N*8),*h_old=(int64_t*)malloc(N*8);
    for(int i=0;i<N;i++){
        h_init[i]=(int64_t)((i-128)*1000003LL);
        h_data[i]=(int64_t)((64-i)*7654321LL);   /* mix so min varies */
    }
    int64_t*d_mem,*d_data,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*8));GPU_CHECK(hipMalloc(&d_data,N*8));GPU_CHECK(hipMalloc(&d_old,N*8));
    GPU_CHECK(hipMemcpy(d_mem,h_init,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_data,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_mem,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int64_t exp_old=h_init[i];
        int64_t exp_mem=h_init[i]<h_data[i]?h_init[i]:h_data[i];
        if(h_old[i]!=exp_old||h_mem[i]!=exp_mem){
            fprintf(stderr,"FAIL[%d] old=%lld(exp %lld) mem=%lld(exp %lld)\n",i,
                (long long)h_old[i],(long long)exp_old,(long long)h_mem[i],(long long)exp_mem);pass=0;break;}
    }
    if(pass)printf("old[1]=%lld mem[1]=%lld\n",(long long)h_old[1],(long long)h_mem[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
