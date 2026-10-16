/*
 * HIP test for V_CVT_I32_F64 (VOP1) plus DPP variant.
 * gem5 (vop1.cc): for finite |value| with frexp exp <= 30,
 *   D.i32 = (int32_t) S0.f64          (truncation toward zero)
 * (out-of-range/inf/nan clamp to INT_MIN/INT_MAX/0; not exercised here.)
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

__global__ void kern_base(const double*a,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s=a[i]; int32_t d;
    __asm__ volatile("v_cvt_i32_f64 %0, %1\n\t":"=v"(d):"v"(s));
    out[i]=d;
}
__global__ void kern_dpp(const double*a,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s=a[i]; int32_t d;
    __asm__ volatile("v_cvt_i32_f64_dpp %0, %1 row_newbcast:1 row_mask:0xf bank_mask:0xf\n\t":"=v"(d):"v"(s));
    out[i]=d;
}
int main(void){
    double*h_a=(double*)malloc(N*8);
    int32_t*h_b=(int32_t*)malloc(N*4),*h_d=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_a[i]=(double)(i-128)*1000.75+0.5;
    double*d_a; int32_t*d_b,*d_d;
    GPU_CHECK(hipMalloc(&d_a,N*8));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_d,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*8,hipMemcpyHostToDevice));
    kern_base<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,N);
    kern_dpp <<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_d,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_b,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_d,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){int32_t eb=(int32_t)h_a[i];if(h_b[i]!=eb){fprintf(stderr,"FAIL base[%d] got %d exp %d\n",i,h_b[i],eb);pass=0;}}
    if(pass)for(int i=0;i<N&&pass;i++){int lane=i&63,row=lane>>4,src=(row<<4)|1,g=(i&~63)|src;int32_t ed=(int32_t)h_a[g];if(h_d[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %d exp %d\n",i,h_d[i],ed);pass=0;}}
    if(pass)printf("base[1]=%d dpp[1]=%d\n",h_b[1],h_d[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
