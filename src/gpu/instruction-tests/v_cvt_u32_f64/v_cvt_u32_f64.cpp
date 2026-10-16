/*
 * HIP test for V_CVT_U32_F64 (VOP1) plus DPP variant.
 * gem5 (vop1.cc): for finite non-negative value with frexp exp <= 31,
 *   D.u32 = (uint32_t) S0.f64          (truncation toward zero)
 * (inf/nan/out-of-range clamp; not exercised here.)
 *
 * DPP: DP-ALU only accepts row_newbcast (lane 1 of each 16-lane row group).
 *
 * Variants: base,dpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kern_base(const double*a,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s=a[i]; uint32_t d;
    __asm__ volatile("v_cvt_u32_f64 %0, %1\n\t":"=v"(d):"v"(s));
    out[i]=d;
}
__global__ void kern_dpp(const double*a,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s=a[i]; uint32_t d;
    __asm__ volatile("v_cvt_u32_f64_dpp %0, %1 row_newbcast:1 row_mask:0xf bank_mask:0xf\n\t":"=v"(d):"v"(s));
    out[i]=d;
}
int main(void){
    double*h_a=(double*)malloc(N*8);
    uint32_t*h_b=(uint32_t*)malloc(N*4),*h_d=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_a[i]=(double)i*10000.5+0.25;   /* non-negative */
    double*d_a; uint32_t*d_b,*d_d;
    GPU_CHECK(hipMalloc(&d_a,N*8));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_d,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*8,hipMemcpyHostToDevice));
    kern_base<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,N);
    kern_dpp <<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_d,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_b,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_d,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){uint32_t eb=(uint32_t)h_a[i];if(h_b[i]!=eb){fprintf(stderr,"FAIL base[%d] got %u exp %u\n",i,h_b[i],eb);pass=0;}}
    if(pass)for(int i=0;i<N&&pass;i++){int lane=i&63,row=lane>>4,src=(row<<4)|1,g=(i&~63)|src;uint32_t ed=(uint32_t)h_a[g];if(h_d[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %u exp %u\n",i,h_d[i],ed);pass=0;}}
    if(pass)printf("base[1]=%u dpp[1]=%u\n",h_b[1],h_d[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
