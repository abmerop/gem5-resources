/*
 * HIP test for V_RSQ_F64 (VOP1, double-precision reciprocal square root).
 * gem5 (vop1.cc). Pattern A (vector), 64-bit operands (register pairs).
 * No DPP/SDWA exercised (f64 DP-ALU DPP requires row_newbcast only).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,        \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void kern(const double *a, double *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s=a[i],d; __asm__ volatile("v_rsq_f64 %0, %1\n\t":"=v"(d):"v"(s)); out[i]=d; }

static double host_ref(double v){ return 1.0/sqrt(v); }
/* The bare V_RSQ_F64 instruction is a reduced-precision hardware
 * approximation (correctly-rounded f64 rsqrt needs a Newton-Raphson refine
 * that the compiler normally emits around it). gem5 models the exact math,
 * so a 1e-5 relative tolerance passes on both gem5 and hardware. */
static int close_d(double g,double e){
    if(isnan(g)&&isnan(e))return 1;
    if(isinf(g)||isinf(e))return (isinf(g)&&isinf(e)&&((g>0)==(e>0)));

    return fabs(g-e) <= 1e-5*(1.0+fabs(e)); }

int main(void){
    size_t sz=N*sizeof(double);
    double *h_a=(double*)malloc(sz),*h_o=(double*)malloc(sz);
    for(int i=0;i<N;i++){ h_a[i]=((double)(i%97)*1.5+0.5); }
    double *d_a,*d_o;
    GPU_CHECK(hipMalloc(&d_a,sz));GPU_CHECK(hipMalloc(&d_o,sz));
    GPU_CHECK(hipMemcpy(d_a,h_a,sz,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,sz,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ double e=host_ref(h_a[i]);
        if(!close_d(h_o[i],e)){fprintf(stderr,"FAIL[%d] in %g got %g exp %g\n",i,h_a[i],h_o[i],e);pass=0;break;} }
    if(pass)printf("out[1]=%g out[%d]=%g\n",h_o[1],N-1,h_o[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_o));free(h_a);free(h_o); return pass?0:1;
}
