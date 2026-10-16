/*
 * HIP test for DS_CMPST_RTN_F32 (ENC_DS, LDS f32 compare-and-store, return).
 *
 * gem5 (ds.cc cmpstExecute<F32>) semantics (DATA0=cmp, DATA1=src):
 *     tmp = MEM[ADDR];
 *     MEM[ADDR] = (tmp == cmp) ? src : tmp;      (f32 equality)
 *     RETURN_DATA = tmp
 * Assembly: ds_cmpst_rtn_f32 vdst, addr, cmp, src.
 * Per-lane distinct slot; returned old == init, final == (init==cmp)?src:init.
 * Half lanes match, half don't; exact-representable f32 values.
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
kern(const float*init,const float*cmp,const float*src,
     float*old_out,float*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ float lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    uint32_t addr=(uint32_t)(uintptr_t)(&lds[t]);
    float c=cmp[i], s=src[i], old;
    __asm__ volatile("ds_cmpst_rtn_f32 %0, %1, %2, %3\n\ts_waitcnt lgkmcnt(0)\n\t"
                     :"=v"(old):"v"(addr),"v"(c),"v"(s):"memory");
    __syncthreads(); old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    float *hi=(float*)malloc(N*4),*hc=(float*)malloc(N*4),
          *hs=(float*)malloc(N*4),*ho=(float*)malloc(N*4),*hm=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ hi[i]=(float)((i%29)-14)*0.5f; hs[i]=(float)((i%5)+1)*16.0f;
        hc[i]=(i&1)?hi[i]:(hi[i]+2.0f); }
    float *di,*dc,*ds,*d_o,*d_m;
    GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&ds,N*4));
    GPU_CHECK(hipMalloc(&d_o,N*4));GPU_CHECK(hipMalloc(&d_m,N*4));
    GPU_CHECK(hipMemcpy(di,hi,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(ds,hs,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dc,ds,d_o,d_m,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,d_o,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hm,d_m,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){ float em=(hi[i]==hc[i])?hs[i]:hi[i];
        if(ho[i]!=hi[i]){fprintf(stderr,"FAIL old[%d] got %g exp %g\n",i,ho[i],hi[i]);pass=0;}
        else if(hm[i]!=em){fprintf(stderr,"FAIL mem[%d] got %g exp %g\n",i,hm[i],em);pass=0;} }
    if(pass)printf("old[0]=%g mem[0]=%g mem[1]=%g\n",ho[0],hm[0],hm[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
