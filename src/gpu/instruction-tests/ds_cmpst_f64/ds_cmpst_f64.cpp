/*
 * HIP test for DS_CMPST_F64 (ENC_DS, LDS 64-bit f64 compare-and-store, no rtn).
 *
 * gem5 (ds.cc cmpstExecute<F64>) semantics (DATA0=cmp, DATA1=src):
 *     tmp = MEM[ADDR];
 *     MEM[ADDR] = (tmp == cmp) ? src : tmp;      (f64 equality, no return)
 * Assembly: ds_cmpst_f64 addr, cmp[0:1], src[0:1].
 * Per-lane distinct 8-byte slot; exact-representable doubles.  Half the lanes
 * match (store src), half don't (keep init).  Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void
kern(const double*init,const double*cmp,const double*src,double*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ double lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    uint32_t addr=(uint32_t)(uintptr_t)(&lds[t]);
    double c=cmp[i], s=src[i];
    __asm__ volatile("ds_cmpst_f64 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t"
                     ::"v"(addr),"v"(c),"v"(s):"memory");
    __syncthreads(); out[i]=lds[t];
}
int main(void){
    double *hi=(double*)malloc(N*8),*hc=(double*)malloc(N*8),
           *hs=(double*)malloc(N*8),*ho=(double*)malloc(N*8);
    for(int i=0;i<N;i++){ hi[i]=(double)((i%31)-15)*0.25; hs[i]=(double)((i%7)+1)*32.0;
        hc[i]=(i&1)?hi[i]:(hi[i]+1.0); }
    double *di,*dc,*ds,*d_o;
    GPU_CHECK(hipMalloc(&di,N*8));GPU_CHECK(hipMalloc(&dc,N*8));
    GPU_CHECK(hipMalloc(&ds,N*8));GPU_CHECK(hipMalloc(&d_o,N*8));
    GPU_CHECK(hipMemcpy(di,hi,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dc,hc,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(ds,hs,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dc,ds,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,d_o,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){ double e=(hi[i]==hc[i])?hs[i]:hi[i];
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %f exp %f\n",i,ho[i],e);pass=0;} }
    if(pass)printf("out[0]=%f out[1]=%f\n",ho[0],ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
