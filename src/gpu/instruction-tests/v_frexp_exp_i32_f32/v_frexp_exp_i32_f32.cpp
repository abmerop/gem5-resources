/*
 * HIP test for V_FREXP_EXP_I32_F32 (VOP1/VOP3, extract binary exponent).
 * gem5 (vop3.cc): if inf/nan -> 0; else std::frexp(src,&exp); vdst = exp
 * (i32).  Host uses std::frexp identically.  Exercises base / dpp / sdwa.
 * Pattern A (vector), one wavefront per block.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kern_base(const float*a,int32_t*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float s=a[i]; int32_t d;
    __asm__ volatile("v_frexp_exp_i32_f32_e32 %0, %1\n\t":"=v"(d):"v"(s)); o[i]=d; }
__global__ void kern_dpp(const float*a,int32_t*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float s=a[i]; int32_t d;
    __asm__ volatile("v_frexp_exp_i32_f32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(d):"v"(s)); o[i]=d; }
__global__ void kern_sdwa(const float*a,int32_t*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float s=a[i]; int32_t d;
    __asm__ volatile("v_frexp_exp_i32_f32_sdwa %0, %1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD\n\t":"=v"(d):"v"(s)); o[i]=d; }

static int32_t host_ref(float x){
    if(isinf(x)||isnan(x))return 0; int e=0; std::frexp(x,&e); return (int32_t)e;
}
int main(void){
    size_t sz=N*sizeof(float),oz=N*sizeof(int32_t);
    float*h_a=(float*)malloc(sz);
    int32_t*h_b=(int32_t*)malloc(oz),*h_d=(int32_t*)malloc(oz),*h_s=(int32_t*)malloc(oz);
    for(int i=0;i<N;i++){ h_a[i]=((float)((i%50)-25))*0.7f + ((i&1)?0.001f:16.0f); if(i%17==0)h_a[i]=0.0f; }
    float*d_a; int32_t*d_b,*d_d,*d_s;
    GPU_CHECK(hipMalloc(&d_a,sz));GPU_CHECK(hipMalloc(&d_b,oz));GPU_CHECK(hipMalloc(&d_d,oz));GPU_CHECK(hipMalloc(&d_s,oz));
    GPU_CHECK(hipMemcpy(d_a,h_a,sz,hipMemcpyHostToDevice));
    int grid=N/BLOCK_SIZE;
    kern_base<<<grid,BLOCK_SIZE>>>(d_a,d_b,N);
    kern_dpp <<<grid,BLOCK_SIZE>>>(d_a,d_d,N);
    kern_sdwa<<<grid,BLOCK_SIZE>>>(d_a,d_s,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_b,oz,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_d,oz,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_s,d_s,oz,hipMemcpyDeviceToHost));
    const int perm[4]={1,0,3,2}; int pass=1;
    for(int i=0;i<N&&pass;i++){
        int32_t eb=host_ref(h_a[i]);
        if(h_b[i]!=eb){fprintf(stderr,"BASE FAIL[%d] in %g got %d exp %d\n",i,h_a[i],h_b[i],eb);pass=0;break;}
        int lane=i%BLOCK_SIZE,base=i-lane; int src=base+((lane&~3)+perm[lane&3]);
        int32_t ed=host_ref(h_a[src]);
        if(h_d[i]!=ed){fprintf(stderr,"DPP FAIL[%d] got %d exp %d\n",i,h_d[i],ed);pass=0;break;}
        if(h_s[i]!=eb){fprintf(stderr,"SDWA FAIL[%d] got %d exp %d\n",i,h_s[i],eb);pass=0;break;}
    }
    if(pass)printf("base[4]=%d dpp[4]=%d sdwa[4]=%d\n",h_b[4],h_d[4],h_s[4]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_d));GPU_CHECK(hipFree(d_s));
    free(h_a);free(h_b);free(h_d);free(h_s); return pass?0:1;
}
