/*
 * HIP test for V_COS_F32 (VOP1, cosine).  Base form only: gem5 (vop1.cc)
 * panic_if(isSDWA)/panic_if(isDPP) for V_COS_F32, so DPP/SDWA are NOT
 * supported and must not be emitted.
 *
 * Hardware/gem5 semantics: D.f = cos(S0.f * 2 * PI) (input in revolutions,
 * math done in double).  Host reference replicates the exact expression.
 * Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void
kern(const float*a,float*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float x=a[i], d;
    __asm__ volatile("v_cos_f32_e32 %0, %1\n\t":"=v"(d):"v"(x));
    o[i]=d;
}
static float cosref(float x){ return (float)cos((double)x * 2.0 * M_PI); }
static int close_f(float g,float e){
    if(isnan(g)&&isnan(e))return 1;
    return fabsf(g-e)<=1e-4f*(1.0f+fabsf(e)); }
int main(void){
    float *ha=(float*)malloc(N*4),*ho=(float*)malloc(N*4);
    for(int i=0;i<N;i++) ha[i]=((float)(i%64)-32.0f)*0.25f;
    float *da,*d_o;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&d_o,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,d_o,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){ float e=cosref(ha[i]);
        if(!close_f(ho[i],e)){fprintf(stderr,"FAIL[%d] in %g got %g exp %g\n",i,ha[i],ho[i],e);pass=0;} }
    if(pass)printf("out[4]=%g\n",ho[4]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
