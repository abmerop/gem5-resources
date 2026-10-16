/*
 * HIP test for FLAT_ATOMIC_MAX_F64 (ENC_FLAT, 64-bit float atomic max, ret-old).
 * Decodes to Inst_FLAT__FLAT_ATOMIC_MAX_F64 (atomicExecute<F64> max); sc0 ->
 * return old.  Each lane owns a private global slot; inputs are finite distinct
 * doubles so max is unambiguous and exact.  old=init, mem=max(init,data).
 * Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(double*mem,const double*data,double*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(&mem[i]);
    double v=data[i], old;
    __asm__ volatile("flat_atomic_max_f64 %0, %1, %2 sc0\n\ts_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
                     : "=v"(old) : "v"(addr), "v"(v) : "memory");
    old_out[i]=old;
}
int main(void){
    double*hi=(double*)malloc(N*8),*hd=(double*)malloc(N*8),*ho=(double*)malloc(N*8),*hm=(double*)malloc(N*8);
    for(int i=0;i<N;i++){ hi[i]=(double)(i-128)*1.5; hd[i]=(double)(64-i)*2.25; }
    double*dm,*dd,*doo;
    GPU_CHECK(hipMalloc(&dm,N*8));GPU_CHECK(hipMalloc(&dd,N*8));GPU_CHECK(hipMalloc(&doo,N*8));
    GPU_CHECK(hipMemcpy(dm,hi,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dd,hd,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dm,dd,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(hm,dm,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(ho,doo,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ double eo=hi[i], em=hi[i]>hd[i]?hi[i]:hd[i];
        if(ho[i]!=eo||hm[i]!=em){fprintf(stderr,"FAIL[%d] old=%g(e %g) mem=%g(e %g)\n",i,ho[i],eo,hm[i],em);pass=0;break;}}
    if(pass)printf("old[0]=%g mem[0]=%g\n",ho[0],hm[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
