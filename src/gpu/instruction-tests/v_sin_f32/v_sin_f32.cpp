/*
 * HIP test for V_SIN_F32 (VOP1/VOP3, sine).
 * Hardware semantics (AMD ISA): D.f = sin(S0.f * 2 * PI), i.e. the input is
 * in revolutions (1.0 == one full turn).  Host reference replicates this
 * exactly (double multiply then double sin) to match both hardware and gem5
 * to within ULP.  Exercises base / dpp quad_perm / sdwa.
 * Pattern A (vector), one wavefront per block.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kern_base(const float*a,float*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float s=a[i],d;
    __asm__ volatile("v_sin_f32_e32 %0, %1\n\t":"=v"(d):"v"(s)); o[i]=d; }
__global__ void kern_dpp(const float*a,float*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float s=a[i],d;
    __asm__ volatile("v_sin_f32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(d):"v"(s)); o[i]=d; }
__global__ void kern_sdwa(const float*a,float*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float s=a[i],d;
    __asm__ volatile("v_sin_f32_sdwa %0, %1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD\n\t":"=v"(d):"v"(s)); o[i]=d; }

static float host_ref(float x){
    return (float)std::sin((double)x * 2.0 * M_PI);
}
static int close_f(float g,float e){
    if(isnan(g)&&isnan(e))return 1;
    return fabsf(g-e)<=1e-4f*(1.0f+fabsf(e)); }

int main(void){
    size_t sz=N*sizeof(float);
    float*h_a=(float*)malloc(sz),*h_b=(float*)malloc(sz),*h_d=(float*)malloc(sz),*h_s=(float*)malloc(sz);
    for(int i=0;i<N;i++){ h_a[i]=((float)(i%64)-32.0f)*0.25f; }
    float*d_a,*d_b,*d_d,*d_s;
    GPU_CHECK(hipMalloc(&d_a,sz));GPU_CHECK(hipMalloc(&d_b,sz));GPU_CHECK(hipMalloc(&d_d,sz));GPU_CHECK(hipMalloc(&d_s,sz));
    GPU_CHECK(hipMemcpy(d_a,h_a,sz,hipMemcpyHostToDevice));
    int grid=N/BLOCK_SIZE;
    kern_base<<<grid,BLOCK_SIZE>>>(d_a,d_b,N);
    kern_dpp <<<grid,BLOCK_SIZE>>>(d_a,d_d,N);
    kern_sdwa<<<grid,BLOCK_SIZE>>>(d_a,d_s,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_b,sz,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_d,sz,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_s,d_s,sz,hipMemcpyDeviceToHost));
    const int perm[4]={1,0,3,2}; int pass=1;
    for(int i=0;i<N&&pass;i++){
        float eb=host_ref(h_a[i]);
        if(!close_f(h_b[i],eb)){fprintf(stderr,"BASE FAIL[%d] in %g got %g exp %g\n",i,h_a[i],h_b[i],eb);pass=0;break;}
        int lane=i%BLOCK_SIZE,base=i-lane; int src=base+((lane&~3)+perm[lane&3]);
        float ed=host_ref(h_a[src]);
        if(!close_f(h_d[i],ed)){fprintf(stderr,"DPP FAIL[%d] got %g exp %g\n",i,h_d[i],ed);pass=0;break;}
        if(!close_f(h_s[i],eb)){fprintf(stderr,"SDWA FAIL[%d] got %g exp %g\n",i,h_s[i],eb);pass=0;break;}
    }
    if(pass)printf("base[4]=%g dpp[4]=%g sdwa[4]=%g\n",h_b[4],h_d[4],h_s[4]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_d));GPU_CHECK(hipFree(d_s));
    free(h_a);free(h_b);free(h_d);free(h_s); return pass?0:1;
}
