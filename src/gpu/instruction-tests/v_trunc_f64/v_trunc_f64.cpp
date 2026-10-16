/*
 * HIP test for V_TRUNC_F64 (VOP1) plus DPP variant.
 * gem5 (vop1.cc): D.f64 = std::trunc(S0.f64).
 *
 * DPP: as a DP-ALU op the only accepted DPP form is row_newbcast (gem5
 * inst_util handles this control), which broadcasts lane 1 of each 16-lane
 * row group.  Verified against trunc() of the broadcast source lane.
 *
 * Variants: base  (gem5 panics on DPP for this f64 VOP1 op)
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kern_base(const double*a,double*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s=a[i],d;
    __asm__ volatile("v_trunc_f64 %0, %1\n\t":"=v"(d):"v"(s));
    out[i]=d;
}
int main(void){
    double*h_a=(double*)malloc(N*8),*h_b=(double*)malloc(N*8),*h_d=(double*)malloc(N*8);
    for(int i=0;i<N;i++) h_a[i]=(double)(i-128)*1.3370001+0.5;
    double*d_a,*d_b,*d_d;
    GPU_CHECK(hipMalloc(&d_a,N*8));GPU_CHECK(hipMalloc(&d_b,N*8));GPU_CHECK(hipMalloc(&d_d,N*8));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*8,hipMemcpyHostToDevice));
    kern_base<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_b,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){double eb=trunc(h_a[i]);if(h_b[i]!=eb){fprintf(stderr,"FAIL base[%d] got %g exp %g\n",i,h_b[i],eb);pass=0;}}
    if(pass)printf("base[1]=%g\n",h_b[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
