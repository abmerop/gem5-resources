/*
 * HIP test for V_CVT_F32_F64 (VOP1 convert double -> float), pattern A.
 * Exercises base and (mandated) DPP variants.
 *
 * gem5 (vop1.cc): D.f32 = f64_to_f32(S0.f64).
 *
 * Variant probing (llvm-mc -mcpu=gfx950):
 *   base : v_cvt_f32_f64_e32 v0, v[2:3]
 *   dpp  : the ONLY accepted DPP form for this DP-ALU op is row_newbcast
 *          ("DP ALU dpp only supports row_newbcast"):
 *            v_cvt_f32_f64_dpp v0, v[2:3] row_newbcast:1 ...
 *
 * This test exercises both base and DPP so the DPP variant can be marked
 * working.  Note: gem5 (vega/insts/inst_util.hh) only implements DPP controls
 * up to ROW_BCAST31 and panics on any other control, so the row_newbcast form
 * is expected to surface a gem5 gap.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_f32_f64 v_cvt_f32_f64.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kern_base(const double*a,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s=a[i]; float d;
    __asm__ volatile("v_cvt_f32_f64 %0, %1\n\t":"=v"(d):"v"(s));
    out[i]=d;
}
__global__ void kern_dpp(const double*a,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s=a[i]; float d;
    __asm__ volatile(
        "v_cvt_f32_f64_dpp %0, %1 row_newbcast:1 row_mask:0xf bank_mask:0xf\n\t"
        :"=v"(d):"v"(s));
    out[i]=d;
}

int main(void){
    double*h_a=(double*)malloc(N*8);
    float*h_b=(float*)malloc(N*4),*h_d=(float*)malloc(N*4);
    for(int i=0;i<N;i++) h_a[i]=(double)(i-128)*1.3370001+0.5;
    double*d_a; float*d_b,*d_d;
    GPU_CHECK(hipMalloc(&d_a,N*8));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_d,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*8,hipMemcpyHostToDevice));
    int grid=N/BLOCK_SIZE;
    kern_base<<<grid,BLOCK_SIZE>>>(d_a,d_b,N);
    kern_dpp <<<grid,BLOCK_SIZE>>>(d_a,d_d,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_b,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_d,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        float eb=(float)h_a[i];
        if(h_b[i]!=eb){fprintf(stderr,"FAIL base[%d] got %g exp %g\n",i,h_b[i],eb);pass=0;break;}
    }
    /* row_newbcast:1 broadcasts lane 1 of each row-group; verify the DPP
     * result matches the converted source lane it selects (lane within row
     * derived from newbcast). If gem5 lacks this control it will panic. */
    if(pass) for(int i=0;i<N;i++){
        int lane=i&63, row=lane>>4, src=(row<<4)|1;        /* lane 1 of row */
        int gsrc=(i&~63)|src;
        float ed=(float)h_a[gsrc];
        if(h_d[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %g exp %g\n",i,h_d[i],ed);pass=0;break;}
    }
    if(pass)printf("base[1]=%g dpp[1]=%g\n",h_b[1],h_d[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
