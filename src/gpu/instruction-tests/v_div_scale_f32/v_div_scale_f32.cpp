/*
 * HIP test for V_DIV_SCALE_F32 (VOP3, div-scale helper).  gem5 only writes vdst
 * deterministically on edge branches; the well-scaled path leaves it unwritten.
 * We drive the src1==0 (FP_ZERO) branch, for which gem5 sets vdst = NaN for every
 * lane (vop3.cc).  The test verifies vdst is NaN.  vcc is a clobbered sdst.
 * Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const float*a,const float*c,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i], s1=0.0f, s2=c[i], r;
    __asm__ volatile("v_div_scale_f32 %0, vcc, %1, %2, %3\n\t"
                     :"=v"(r):"v"(s0),"v"(s1),"v"(s2):"vcc");
    out[i]=r;
}
int main(void){
    float*ha=(float*)malloc(N*4),*hc=(float*)malloc(N*4),*ho=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ha[i]=(float)(i+1)*0.5f;hc[i]=(float)(i%13)+1.0f;}
    float*da,*dc,*dO; GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,dc,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ if(!isnan(ho[i])){fprintf(stderr,"FAIL[%d] got %g exp NaN\n",i,ho[i]);pass=0;break;} }
    if(pass)printf("all lanes NaN (src1==0 branch)\n");
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
