/*
 * HIP test for V_CVT_RPI_I32_F32 (VOP1, round-to-nearest (ties up) f32->i32).
 * gem5 (vop1.cc): D.i32 = (int)std::floor(S0.f32 + 0.5) via vop1CvtHelper
 * (DPP/SDWA both supported). Inputs are moderate finite floats so the
 * conversion stays well within int32 range.
 * Exercises base / dpp quad_perm:[1,0,3,2] / sdwa. Pattern A (vector).
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

__global__ void kern_base(const float *a, int32_t *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s=a[i]; int32_t d; __asm__ volatile("v_cvt_rpi_i32_f32_e32 %0, %1\n\t":"=v"(d):"v"(s)); out[i]=d; }
__global__ void kern_dpp(const float *a, int32_t *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s=a[i]; int32_t d; __asm__ volatile("v_cvt_rpi_i32_f32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(d):"v"(s)); out[i]=d; }
__global__ void kern_sdwa(const float *a, int32_t *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s=a[i]; int32_t d; __asm__ volatile("v_cvt_rpi_i32_f32_sdwa %0, %1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD\n\t":"=v"(d):"v"(s)); out[i]=d; }

static int32_t host_ref(float x){ return (int32_t)floor((double)x + 0.5); }

int main(void){
    size_t szf=N*sizeof(float), szi=N*sizeof(int32_t);
    float *h_a=(float*)malloc(szf);
    int32_t *h_b=(int32_t*)malloc(szi),*h_d=(int32_t*)malloc(szi),*h_s=(int32_t*)malloc(szi);
    for(int i=0;i<N;i++) h_a[i]=((float)((i*97)%4000)-2000.0f)*0.5f + 0.25f*((i&1)?1:-1);
    float *d_a; int32_t *d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,szf));GPU_CHECK(hipMalloc(&d_base,szi));GPU_CHECK(hipMalloc(&d_dpp,szi));GPU_CHECK(hipMalloc(&d_sdwa,szi));
    GPU_CHECK(hipMemcpy(d_a,h_a,szf,hipMemcpyHostToDevice));
    int grid=N/BLOCK_SIZE;
    kern_base<<<grid,BLOCK_SIZE>>>(d_a,d_base,N);
    kern_dpp <<<grid,BLOCK_SIZE>>>(d_a,d_dpp ,N);
    kern_sdwa<<<grid,BLOCK_SIZE>>>(d_a,d_sdwa,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_base,szi,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_dpp ,szi,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_s,d_sdwa,szi,hipMemcpyDeviceToHost));
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
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_base));GPU_CHECK(hipFree(d_dpp));GPU_CHECK(hipFree(d_sdwa));
    free(h_a);free(h_b);free(h_d);free(h_s); return pass?0:1;
}
