/*
 * HIP test for V_FREXP_MANT_F64 (VOP1, f64 frexp mantissa).
 * gem5 (vop1.cc): if inf/nan -> vdst = src; else vdst = frexp(src, &exp),
 * i.e. the normalized mantissa in [0.5, 1) (sign preserved).
 * gem5 panics on DPP/SDWA, so base (e32) only.
 * Inputs are exact (finite) doubles -> frexp mantissa is exact -> bit-exact.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kern(const double*a,double*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s0=a[i],d;
    __asm__ volatile("v_frexp_mant_f64_e32 %0, %1\n\t":"=v"(d):"v"(s0));
    out[i]=d;
}

int main(void){
    double*ha=(double*)malloc(N*8),*ho=(double*)malloc(N*8);
    for(int i=0;i<N;i++){
        // exact finite doubles of varied magnitude and sign, incl 0
        double v = (double)((i%17)-8) * ((i%3)==0 ? 0.25 : (i%3)==1 ? 8.0 : 1.0);
        if(i==0) v=0.0; else if(i==5) v=6.0; else if(i==6) v=-40.0;
        ha[i]=v;
    }
    double*da,*doo; GPU_CHECK(hipMalloc(&da,N*8));GPU_CHECK(hipMalloc(&doo,N*8));
    GPU_CHECK(hipMemcpy(da,ha,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ int e2; double e=frexp(ha[i],&e2);
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] in=%g got %g exp %g\n",i,ha[i],ho[i],e);pass=0;break;} }
    if(pass)printf("out[5]=%g out[6]=%g\n",ho[5],ho[6]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
