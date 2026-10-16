/*
 * HIP test for V_CVT_F32_UBYTE2 (VOP1 convert unsigned byte-2 to f32), with
 * DPP and SDWA.
 *
 * gem5 (vop1.cc via vop1CvtHelper): vdst = (float)bits(src, 23, 16) (byte 2).
 *
 *   base : v_cvt_f32_ubyte2_e32 vd, va        (float)((a[i]>>16)&0xff)
 *   dpp  : quad_perm:[1,0,3,2] swizzles src0 -> byte2 of a[swizzled_lane]
 *   sdwa : src0_sel:DWORD (identity)          (float)((a[i]>>16)&0xff)
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_f32_ubyte2 v_cvt_f32_ubyte2.cpp
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
#define SHIFT 16

__global__ void
kbase(const uint32_t *a, float *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; uint32_t s0=a[i]; float vd;
    __asm__ volatile("v_cvt_f32_ubyte2_e32 %0, %1\n\t" : "=v"(vd) : "v"(s0)); out[i]=vd;
}
__global__ void
kdpp(const uint32_t *a, float *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; uint32_t s0=a[i]; float vd;
    __asm__ volatile("v_cvt_f32_ubyte2_dpp %0, %1 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t" : "=v"(vd) : "v"(s0)); out[i]=vd;
}
__global__ void
ksdwa(const uint32_t *a, float *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; uint32_t s0=a[i]; float vd;
    __asm__ volatile("v_cvt_f32_ubyte2_sdwa %0, %1 dst_sel:DWORD "
                     "dst_unused:UNUSED_PAD src0_sel:DWORD\n\t" : "=v"(vd) : "v"(s0)); out[i]=vd;
}

int main(void) {
    uint32_t *h_a=(uint32_t*)malloc(N*4);
    float *h_base=(float*)malloc(N*4),*h_dpp=(float*)malloc(N*4),*h_sdwa=(float*)malloc(N*4);
    for (int i=0;i<N;i++) h_a[i]=0x9e3779b9u*(uint32_t)(i+1);
    uint32_t *d_a; float *d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_base,N*4));
    GPU_CHECK(hipMalloc(&d_dpp,N*4));GPU_CHECK(hipMalloc(&d_sdwa,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    int g=N/BLOCK_SIZE;
    hipLaunchKernelGGL(kbase,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_base,N);GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kdpp,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_dpp,N);GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(ksdwa,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_sdwa,N);GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base,d_base,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp,d_dpp,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa,d_sdwa,N*4,hipMemcpyDeviceToHost));
    static const int perm[4]={1,0,3,2}; int pass=1;
    for (int i=0;i<N;i++){
        float eb=(float)((h_a[i]>>SHIFT)&0xffu);
        if (h_base[i]!=eb){fprintf(stderr,"FAIL base[%d] got %g exp %g\n",i,h_base[i],eb);pass=0;break;}
        int lane=i%BLOCK_SIZE,wb=i-lane,sl=(lane&~3)+perm[lane&3];
        float ed=(float)((h_a[wb+sl]>>SHIFT)&0xffu);
        if (h_dpp[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %g exp %g\n",i,h_dpp[i],ed);pass=0;break;}
        float es=(float)((h_a[i]>>SHIFT)&0xffu);
        if (h_sdwa[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got %g exp %g\n",i,h_sdwa[i],es);pass=0;break;}
    }
    if (pass) printf("base[1]=%g dpp[1]=%g sdwa[1]=%g\n",h_base[1],h_dpp[1],h_sdwa[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
