#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);}} while(0)

static const int N=256, BLK=64;
__global__ void k(const float*a,const float*b,float*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float va=a[i],vb=b[i],r;
    __asm__ volatile("v_fmamk_f32 %0, %1, 0x40000000, %2":"=v"(r):"v"(va),"v"(vb)); o[i]=r;
}
int main(void){
    float*ha=(float*)malloc(N*4),*hb=(float*)malloc(N*4),*ho=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ha[i]=(float)(i%7-3);hb[i]=(float)(i%5+1);}
    float*da,*db,*dbuf; GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dbuf,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k,dim3(N/BLK),dim3(BLK),0,0,da,db,dbuf,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dbuf,N*4,hipMemcpyDeviceToHost));
    int pass=1; for(int i=0;i<N&&pass;i++){ float e=ha[i]*2.0f+hb[i]; if(fabsf(ho[i]-e)>1e-4f){fprintf(stderr,"FAIL[%d] %.3f!=%.3f\n",i,ho[i],e);pass=0;}}
    printf("v_fmamk_f32 d=a*2+b\n"); printf("%s\n",pass?"PASSED":"FAILED"); return pass?0:1;
}
