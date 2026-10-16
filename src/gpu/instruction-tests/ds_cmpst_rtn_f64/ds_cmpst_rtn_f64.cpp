/*
 * HIP test for DS_CMPST_RTN_F64 (ENC_DS, LDS 64-bit f64 compare-and-store, rtn).
 *
 * gem5 (ds.cc cmpstExecute<F64>) semantics (DATA0=cmp, DATA1=src):
 *     tmp = MEM[ADDR];
 *     MEM[ADDR] = (tmp == cmp) ? src : tmp;      (f64 equality)
 *     RETURN_DATA = tmp
 * Assembly: ds_cmpst_rtn_f64 vdst[0:1], addr, cmp[0:1], src[0:1].
 * Per-lane distinct 8-byte slot; old==init, final=(init==cmp)?src:init.
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
kern(const double*init,const double*cmp,const double*src,
     double*old_out,double*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ double lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    uint32_t addr=(uint32_t)(uintptr_t)(&lds[t]);
    double c=cmp[i], s=src[i], old;
    __asm__ volatile("ds_cmpst_rtn_f64 %0, %1, %2, %3\n\ts_waitcnt lgkmcnt(0)\n\t"
                     :"=v"(old):"v"(addr),"v"(c),"v"(s):"memory");
    __syncthreads(); old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    double *hi=(double*)malloc(N*8),*hc=(double*)malloc(N*8),
           *hs=(double*)malloc(N*8),*ho=(double*)malloc(N*8),*hm=(double*)malloc(N*8);
    for(int i=0;i<N;i++){ hi[i]=(double)((i%29)-14)*0.5; hs[i]=(double)((i%5)+1)*64.0;
        hc[i]=(i&1)?hi[i]:(hi[i]+2.0); }
    double *di,*dc,*ds,*d_o,*d_m;
    GPU_CHECK(hipMalloc(&di,N*8));GPU_CHECK(hipMalloc(&dc,N*8));GPU_CHECK(hipMalloc(&ds,N*8));
    GPU_CHECK(hipMalloc(&d_o,N*8));GPU_CHECK(hipMalloc(&d_m,N*8));
    GPU_CHECK(hipMemcpy(di,hi,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dc,hc,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(ds,hs,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dc,ds,d_o,d_m,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,d_o,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hm,d_m,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){ double em=(hi[i]==hc[i])?hs[i]:hi[i];
        if(ho[i]!=hi[i]){fprintf(stderr,"FAIL old[%d] got %f exp %f\n",i,ho[i],hi[i]);pass=0;}
        else if(hm[i]!=em){fprintf(stderr,"FAIL mem[%d] got %f exp %f\n",i,hm[i],em);pass=0;} }
    if(pass)printf("old[0]=%f mem[0]=%f mem[1]=%f\n",ho[0],hm[0],hm[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
