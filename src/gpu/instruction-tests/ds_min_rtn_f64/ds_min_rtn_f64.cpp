/*
 * HIP test for DS_MIN_RTN_F64 (ENC_DS, 64-bit LDS float min atomic, ret-old).
 *   old = LDS[addr]; LDS[addr] = min(old, DATA) (f64); VDST = old.
 * gem5 (ds.cc) sets AtomicMin/AtomicReturn/F64.  Each lane owns a private LDS
 * slot (no contention); inputs are finite, distinct doubles so min is
 * unambiguous and exact.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const double*init,const double*data,double*out_old,double*out_new,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ double lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    uint32_t addr=(uint32_t)(uintptr_t)(__attribute__((address_space(3)))double*)(&lds[t]);
    double v=data[i], old;
    __asm__ volatile("ds_min_rtn_f64 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t"
                     : "=v"(old) : "v"(addr), "v"(v) : "memory");
    __syncthreads();
    out_old[i]=old; out_new[i]=lds[t];
}
int main(void){
    double*hi=(double*)malloc(N*8),*hd=(double*)malloc(N*8),*ho=(double*)malloc(N*8),*hn=(double*)malloc(N*8);
    for(int i=0;i<N;i++){ hi[i]=(double)(i-128)*1.25; hd[i]=(double)(64-i)*2.5; }
    double*di,*dd,*doo,*dn;
    GPU_CHECK(hipMalloc(&di,N*8));GPU_CHECK(hipMalloc(&dd,N*8));
    GPU_CHECK(hipMalloc(&doo,N*8));GPU_CHECK(hipMalloc(&dn,N*8));
    GPU_CHECK(hipMemcpy(di,hi,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dd,hd,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,doo,dn,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hn,dn,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ double eo=hi[i], en=hi[i]<hd[i]?hi[i]:hd[i];
        if(ho[i]!=eo||hn[i]!=en){fprintf(stderr,"FAIL[%d] old=%g(e %g) new=%g(e %g)\n",i,ho[i],eo,hn[i],en);pass=0;break;}}
    if(pass)printf("old[0]=%g new[0]=%g\n",ho[0],hn[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
