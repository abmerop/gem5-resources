/*
 * HIP test for V_CVT_F32_UBYTE0 (VOP1, convert byte 0 of src to f32).
 * gem5 (vop1.cc): D.f32 = (float)(S0[7:0]) via vop1CvtHelper (DPP/SDWA both
 * supported). With sdwa src0_sel:DWORD the whole dword is read then byte 0
 * extracted, matching the base result.
 * Exercises base / dpp quad_perm:[1,0,3,2] / sdwa. Pattern A (vector).
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

__global__ void kern_base(const uint32_t *a, float *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s=a[i]; float d; __asm__ volatile("v_cvt_f32_ubyte0_e32 %0, %1\n\t":"=v"(d):"v"(s)); out[i]=d; }
__global__ void kern_dpp(const uint32_t *a, float *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s=a[i]; float d; __asm__ volatile("v_cvt_f32_ubyte0_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(d):"v"(s)); out[i]=d; }
__global__ void kern_sdwa(const uint32_t *a, float *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s=a[i]; float d; __asm__ volatile("v_cvt_f32_ubyte0_sdwa %0, %1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD\n\t":"=v"(d):"v"(s)); out[i]=d; }

int main(void){
    size_t szu=N*sizeof(uint32_t), szf=N*sizeof(float);
    uint32_t *h_a=(uint32_t*)malloc(szu);
    float *h_b=(float*)malloc(szf),*h_d=(float*)malloc(szf),*h_s=(float*)malloc(szf);
    for(int i=0;i<N;i++) h_a[i]=(uint32_t)(i*2654435761u + 0x11223300u);
    uint32_t *d_a; float *d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,szu));GPU_CHECK(hipMalloc(&d_base,szf));GPU_CHECK(hipMalloc(&d_dpp,szf));GPU_CHECK(hipMalloc(&d_sdwa,szf));
    GPU_CHECK(hipMemcpy(d_a,h_a,szu,hipMemcpyHostToDevice));
    int grid=N/BLOCK_SIZE;
    kern_base<<<grid,BLOCK_SIZE>>>(d_a,d_base,N);
    kern_dpp <<<grid,BLOCK_SIZE>>>(d_a,d_dpp ,N);
    kern_sdwa<<<grid,BLOCK_SIZE>>>(d_a,d_sdwa,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_base,szf,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_dpp ,szf,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_s,d_sdwa,szf,hipMemcpyDeviceToHost));
    const int perm[4]={1,0,3,2}; int pass=1;
    for(int i=0;i<N&&pass;i++){
        float eb=(float)(h_a[i]&0xFFu);
        if(h_b[i]!=eb){fprintf(stderr,"BASE FAIL[%d] got %g exp %g\n",i,h_b[i],eb);pass=0;break;}
        int lane=i%BLOCK_SIZE,base=i-lane; int src=base+((lane&~3)+perm[lane&3]);
        float ed=(float)(h_a[src]&0xFFu);
        if(h_d[i]!=ed){fprintf(stderr,"DPP FAIL[%d] got %g exp %g\n",i,h_d[i],ed);pass=0;break;}
        if(h_s[i]!=eb){fprintf(stderr,"SDWA FAIL[%d] got %g exp %g\n",i,h_s[i],eb);pass=0;break;}
    }
    if(pass)printf("base[4]=%g dpp[4]=%g sdwa[4]=%g\n",h_b[4],h_d[4],h_s[4]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_base));GPU_CHECK(hipFree(d_dpp));GPU_CHECK(hipFree(d_sdwa));
    free(h_a);free(h_b);free(h_d);free(h_s); return pass?0:1;
}
