/*
 * HIP test for V_SUBREV_F32 (VOP2 reversed f32 subtract), with DPP and SDWA.
 *
 * gem5 (vop2.cc via vop2Helper): vdst = src1 - src0.  The generic vop2Helper
 * applies the DPP swizzle / SDWA src-select to SRC0.  Inputs are small
 * integer-valued floats so the subtraction is exact.
 *
 *   base : v_subrev_f32_e32 vd, va, vb        vd = b[i] - a[i]
 *   dpp  : quad_perm:[1,0,3,2] swizzles src0=a -> vd = b[i] - a[swizzled]
 *   sdwa : src0_sel:DWORD/src1_sel:DWORD (identity) -> vd = b[i] - a[i]
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_subrev_f32 v_subrev_f32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kbase(const float *a, const float *b, float *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i],s1=b[i],vd;
    __asm__ volatile("v_subrev_f32_e32 %0, %1, %2\n\t":"=v"(vd):"v"(s0),"v"(s1));
    out[i]=vd;
}
__global__ void
kdpp(const float *a, const float *b, float *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i],s1=b[i],vd;
    __asm__ volatile("v_subrev_f32_dpp %0, %1, %2 quad_perm:[1,0,3,2] "
        "row_mask:0xf bank_mask:0xf\n\t":"=v"(vd):"v"(s0),"v"(s1));
    out[i]=vd;
}
__global__ void
ksdwa(const float *a, const float *b, float *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i],s1=b[i],vd;
    __asm__ volatile("v_subrev_f32_sdwa %0, %1, %2 dst_sel:DWORD "
        "dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD\n\t"
        :"=v"(vd):"v"(s0),"v"(s1));
    out[i]=vd;
}

int main(void){
    float *h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4);
    float *h_base=(float*)malloc(N*4),*h_dpp=(float*)malloc(N*4),*h_sdwa=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ h_a[i]=(float)((i%97)-48); h_b[i]=(float)((i%61)-30)*2.0f; }
    float *d_a,*d_b,*d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_base,N*4));GPU_CHECK(hipMalloc(&d_dpp,N*4));GPU_CHECK(hipMalloc(&d_sdwa,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    int g=N/BLOCK_SIZE;
    hipLaunchKernelGGL(kbase,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_base,N);GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kdpp,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_dpp,N);GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(ksdwa,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_sdwa,N);GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base,d_base,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp,d_dpp,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa,d_sdwa,N*4,hipMemcpyDeviceToHost));
    static const int perm[4]={1,0,3,2}; int pass=1;
    for(int i=0;i<N;i++){
        float eb=h_b[i]-h_a[i];
        if(h_base[i]!=eb){fprintf(stderr,"FAIL base[%d] got %g exp %g\n",i,h_base[i],eb);pass=0;break;}
        int lane=i%BLOCK_SIZE,wb=i-lane,sl=(lane&~3)+perm[lane&3];
        float ed=h_b[i]-h_a[wb+sl];
        if(h_dpp[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %g exp %g\n",i,h_dpp[i],ed);pass=0;break;}
        float es=h_b[i]-h_a[i];
        if(h_sdwa[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got %g exp %g\n",i,h_sdwa[i],es);pass=0;break;}
    }
    if(pass)printf("base[1]=%g dpp[1]=%g sdwa[1]=%g\n",h_base[1],h_dpp[1],h_sdwa[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
