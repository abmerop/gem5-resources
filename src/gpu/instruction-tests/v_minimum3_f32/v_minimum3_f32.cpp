/*
 * HIP test for V_MINIMUM3_F32 (ENC_VOP3), CDNA4 / gfx950.
 * gem5 semantics: D.f = min3(S0,S1,S2), NaN-propagating (dx10_clamp off here).
 * Uses all-positive finite inputs (no NaN) so result == std::min of the three.
 * (The vdst-type bug was fixed by commit f18dd9b805.)
 * Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N=256, BLOCK_SIZE=64;
__global__ void
kern(const float *a,const float *b,const float *c,float *o,int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float x=a[i],y=b[i],z=c[i],vd;
    __asm__ volatile("v_minimum3_f32 %0, %1, %2, %3\n\t":"=v"(vd):"v"(x),"v"(y),"v"(z));
    o[i]=vd;
}
int main(void)
{
    float *h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4),*h_c=(float*)malloc(N*4),*h_o=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ h_a[i]=(float)((i*3)%50+1); h_b[i]=(float)((i*7)%50+1); h_c[i]=(float)((i*13)%50+1); }
    float *d_a,*d_b,*d_c,*d_o;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_o,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){
        float exp=fminf(fminf(h_a[i],h_b[i]),h_c[i]);
        if(h_o[i]!=exp){fprintf(stderr,"FAIL[%d] got %f exp %f\n",i,h_o[i],exp);pass=0;}
    }
    if(pass)printf("o[0]=%f o[1]=%f\n",h_o[0],h_o[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_o));
    free(h_a);free(h_b);free(h_c);free(h_o); return pass?0:1;
}
