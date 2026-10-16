/*
 * HIP test for V_FLOOR_F64 (VOP1, f64 floor).
 * gem5 (vop1.cc): vdst = std::floor(src) (panics on DPP/SDWA, so base only).
 * Inputs are multiples of 0.25 -> floor is exact -> bit-exact compare.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const double*a,double*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s0=a[i],d;
    __asm__ volatile("v_floor_f64_e32 %0, %1\n\t":"=v"(d):"v"(s0));
    out[i]=d;
}
int main(void){
    double*ha=(double*)malloc(N*8),*ho=(double*)malloc(N*8);
    for(int i=0;i<N;i++) ha[i]=((double)(i-128))*0.25;
    double*da,*doo; GPU_CHECK(hipMalloc(&da,N*8));GPU_CHECK(hipMalloc(&doo,N*8));
    GPU_CHECK(hipMemcpy(da,ha,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ double e=floor(ha[i]); if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,ho[i],e);pass=0;break;}}
    if(pass)printf("out[0]=%g out[200]=%g\n",ho[0],ho[200]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
