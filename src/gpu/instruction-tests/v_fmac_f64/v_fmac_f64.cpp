/*
 * HIP test for V_FMAC_F64 (VOP3, fused multiply-accumulate f64).
 * gem5 (vop3.cc): vdst = fma(src0, src1, vdst) (accumulate into dst; panics on
 * DPP so base only).  Small exact integers -> a*b+c exact -> bit-exact compare.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const double*a,const double*b,const double*c,double*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s0=a[i],s1=b[i],d=c[i];
    __asm__ volatile("v_fmac_f64 %0, %1, %2\n\t":"+v"(d):"v"(s0),"v"(s1));
    out[i]=d;
}
int main(void){
    double*ha=(double*)malloc(N*8),*hb=(double*)malloc(N*8),*hc=(double*)malloc(N*8),*ho=(double*)malloc(N*8);
    for(int i=0;i<N;i++){ ha[i]=(double)((i%13)-6); hb[i]=(double)((i%9)+1); hc[i]=(double)((i%5)*4-8); }
    double*da,*db,*dc,*doo;
    GPU_CHECK(hipMalloc(&da,N*8));GPU_CHECK(hipMalloc(&db,N*8));GPU_CHECK(hipMalloc(&dc,N*8));GPU_CHECK(hipMalloc(&doo,N*8));
    GPU_CHECK(hipMemcpy(da,ha,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ double e=ha[i]*hb[i]+hc[i]; if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,ho[i],e);pass=0;break;}}
    if(pass)printf("out[0]=%g out[1]=%g\n",ho[0],ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
