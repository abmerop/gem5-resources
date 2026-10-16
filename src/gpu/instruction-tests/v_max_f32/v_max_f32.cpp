/*
 * HIP test for V_MAX_F32 (VOP2 single-precision maximum), with DPP and SDWA.
 *
 *   D.f32 = std::fmax(S0.f32, S1.f32)
 *
 * gem5 dispatches the VOP2 DPP/SDWA forms through vop2Helper.  Inputs are small
 * integer-valued floats (exact), mixed sign.
 *
 * Three kernels, each self-checked:
 *   base : v_max_f32_e64 vd, va, vb                 -> max(a, b)
 *   dpp  : v_max_f32_dpp vd, va, vb quad_perm:[1,0,3,2] ...
 *            src0 gathered from quadBase+perm[lane&3] -> max(a[swz], b)
 *   sdwa : v_max_f32_sdwa vd, va, vb dst_sel:DWORD dst_unused:UNUSED_PAD
 *            src0_sel:DWORD src1_sel:DWORD          -> max(a, b) (identity)
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_max_f32 v_max_f32.cpp
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

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kernel_base(const float *a, const float *b, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float va=a[i], vb=b[i], vd;
    __asm__ volatile("v_max_f32_e64 %0, %1, %2\n\t" : "=v"(vd) : "v"(va), "v"(vb));
    out[i]=vd;
}
__global__ void
kernel_dpp(const float *a, const float *b, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float va=a[i], vb=b[i], vd;
    __asm__ volatile("v_max_f32_dpp %0, %1, %2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb));
    out[i]=vd;
}
__global__ void
kernel_sdwa(const float *a, const float *b, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float va=a[i], vb=b[i], vd;
    __asm__ volatile("v_max_f32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb));
    out[i]=vd;
}

int main(void)
{
    float *h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4);
    float *h_base=(float*)malloc(N*4),*h_dpp=(float*)malloc(N*4),*h_sdwa=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ h_a[i]=(float)((i%23)-11); h_b[i]=(float)((i%13)-6); }
    float *d_a,*d_b,*d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_base,N*4));GPU_CHECK(hipMalloc(&d_dpp,N*4));GPU_CHECK(hipMalloc(&d_sdwa,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    int grid=N/BLOCK_SIZE;
    kernel_base<<<grid,BLOCK_SIZE>>>(d_a,d_b,d_base,N); GPU_CHECK(hipGetLastError());
    kernel_dpp <<<grid,BLOCK_SIZE>>>(d_a,d_b,d_dpp ,N); GPU_CHECK(hipGetLastError());
    kernel_sdwa<<<grid,BLOCK_SIZE>>>(d_a,d_b,d_sdwa,N); GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base,d_base,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp ,d_dpp ,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa,d_sdwa,N*4,hipMemcpyDeviceToHost));
    static const int perm[4]={1,0,3,2};
    int pass=1;
    for(int i=0;i<N && pass;i++){
        float eb=fmaxf(h_a[i],h_b[i]);
        int bl=(i & ~3)+perm[i&3];
        float ed=fmaxf(h_a[bl],h_b[i]);
        float es=fmaxf(h_a[i],h_b[i]);
        if(h_base[i]!=eb){fprintf(stderr,"BASE FAIL[%d] %f exp %f\n",i,h_base[i],eb);pass=0;}
        else if(h_dpp[i]!=ed){fprintf(stderr,"DPP FAIL[%d] %f exp %f\n",i,h_dpp[i],ed);pass=0;}
        else if(h_sdwa[i]!=es){fprintf(stderr,"SDWA FAIL[%d] %f exp %f\n",i,h_sdwa[i],es);pass=0;}
    }
    if(pass)printf("base[0]=%f dpp[0]=%f sdwa[0]=%f\n",h_base[0],h_dpp[0],h_sdwa[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
