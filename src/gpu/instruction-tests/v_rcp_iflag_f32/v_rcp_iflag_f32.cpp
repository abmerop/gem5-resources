/*
 * HIP test for V_RCP_IFLAG_F32 (VOP1 reciprocal w/ integer-divide flag),
 * pattern A.  Exercises base, DPP and SDWA encodings.
 *
 * gem5 (vop1.cc): D.f32 = 1.0 / S0.f32 (computed exactly in the model).
 *
 * Variants exercised (probed with llvm-mc -mcpu=gfx950):
 *   base : v_rcp_iflag_f32        vdst, vsrc
 *   dpp  : v_rcp_iflag_f32_dpp    vdst, vsrc quad_perm:[1,0,3,2] ...
 *            -> each lane reads its pair-swapped neighbour (lane ^ 1)
 *   sdwa : v_rcp_iflag_f32_sdwa   vdst, vsrc dst_sel:DWORD src0_sel:DWORD
 *            -> full-dword select, behaves like the base form
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_rcp_iflag_f32 v_rcp_iflag_f32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
static const float RTOL=1e-3f;

__global__ void kern_base(const float*a,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s=a[i],d;
    __asm__ volatile("v_rcp_iflag_f32 %0, %1\n\t":"=v"(d):"v"(s));
    out[i]=d;
}
__global__ void kern_dpp(const float*a,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s=a[i],d;
    __asm__ volatile(
        "v_rcp_iflag_f32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t"
        :"=v"(d):"v"(s));
    out[i]=d;
}
__global__ void kern_sdwa(const float*a,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s=a[i],d;
    __asm__ volatile(
        "v_rcp_iflag_f32_sdwa %0, %1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD\n\t"
        :"=v"(d):"v"(s));
    out[i]=d;
}

static int close(float got,float exp){
    return fabsf(got-exp) <= RTOL*fabsf(exp)+1e-6f;
}

int main(void){
    float*h_a=(float*)malloc(N*4);
    float*h_b=(float*)malloc(N*4),*h_d=(float*)malloc(N*4),*h_s=(float*)malloc(N*4);
    for(int i=0;i<N;i++) h_a[i]=(float)(i+1)*0.25f+0.5f;   /* positive non-zero */
    float*d_a,*d_b,*d_d,*d_s;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_d,N*4));GPU_CHECK(hipMalloc(&d_s,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    int grid=N/BLOCK_SIZE;
    kern_base<<<grid,BLOCK_SIZE>>>(d_a,d_b,N);
    kern_dpp <<<grid,BLOCK_SIZE>>>(d_a,d_d,N);
    kern_sdwa<<<grid,BLOCK_SIZE>>>(d_a,d_s,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_b,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_d,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_s,d_s,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N && pass;i++){
        float eb=1.0f/h_a[i];
        float ed=1.0f/h_a[i^1];   /* quad_perm pair swap */
        float es=1.0f/h_a[i];
        if(!close(h_b[i],eb)){fprintf(stderr,"FAIL base[%d] got %g exp %g\n",i,h_b[i],eb);pass=0;break;}
        if(!close(h_d[i],ed)){fprintf(stderr,"FAIL dpp[%d] got %g exp %g\n",i,h_d[i],ed);pass=0;break;}
        if(!close(h_s[i],es)){fprintf(stderr,"FAIL sdwa[%d] got %g exp %g\n",i,h_s[i],es);pass=0;break;}
    }
    if(pass)printf("base[1]=%g dpp[1]=%g sdwa[1]=%g\n",h_b[1],h_d[1],h_s[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
