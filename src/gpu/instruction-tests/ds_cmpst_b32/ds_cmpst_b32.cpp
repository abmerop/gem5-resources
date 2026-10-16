/*
 * HIP test for DS_CMPST_B32 (ENC_DS, LDS 32-bit compare-and-store, no return).
 *
 * gem5 (ds.cc cmpstExecute<U32>) semantics (DATA0=cmp, DATA1=src):
 *     tmp = MEM[ADDR];
 *     MEM[ADDR] = (tmp == cmp) ? src : tmp;      (no value returned)
 * Assembly operand order: ds_cmpst_b32 addr, cmp, src.
 * Each lane owns a distinct 4-byte LDS slot.  Half the lanes use cmp==init
 * (store src) and half use cmp!=init (keep init), covering both branches.
 * Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void
kern(const uint32_t*init,const uint32_t*cmp,const uint32_t*src,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    uint32_t addr=(uint32_t)(uintptr_t)(&lds[t]);
    uint32_t c=cmp[i], s=src[i];
    __asm__ volatile("ds_cmpst_b32 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t"
                     ::"v"(addr),"v"(c),"v"(s):"memory");
    __syncthreads(); out[i]=lds[t];
}
int main(void){
    uint32_t *hi=(uint32_t*)malloc(N*4),*hc=(uint32_t*)malloc(N*4),
             *hs=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ hi[i]=0x1000u+i*7u; hs[i]=0xABCD0000u+i;
        hc[i]=(i&1)?hi[i]:(hi[i]^0xFFu); /* even: match, odd: no match */ }
    uint32_t *di,*dc,*ds,*d_o;
    GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dc,N*4));
    GPU_CHECK(hipMalloc(&ds,N*4));GPU_CHECK(hipMalloc(&d_o,N*4));
    GPU_CHECK(hipMemcpy(di,hi,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(ds,hs,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dc,ds,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,d_o,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){ uint32_t e=(hi[i]==hc[i])?hs[i]:hi[i];
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,ho[i],e);pass=0;} }
    if(pass)printf("out[0]=%08x out[1]=%08x\n",ho[0],ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
