#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);}} while(0)

typedef float f2 __attribute__((ext_vector_type(2)));
static const int N=256, BLK=64;
__global__ void k(const f2*a,const f2*b,f2*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    f2 va=a[i],vb=b[i],r;
    __asm__ volatile("v_pk_mul_f32 %0, %1, %2":"=v"(r):"v"(va),"v"(vb)); o[i]=r;
}
int main(void){
    f2*ha=(f2*)malloc(N*8),*hb=(f2*)malloc(N*8),*ho=(f2*)malloc(N*8);
    for(int i=0;i<N;i++){ha[i][0]=(float)(i%7-3);ha[i][1]=(float)(i%9-4);hb[i][0]=(float)(i%5+1);hb[i][1]=(float)(i%3+1);}
    f2*da,*db,*dbuf; GPU_CHECK(hipMalloc(&da,N*8));GPU_CHECK(hipMalloc(&db,N*8));GPU_CHECK(hipMalloc(&dbuf,N*8));
    GPU_CHECK(hipMemcpy(da,ha,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*8,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k,dim3(N/BLK),dim3(BLK),0,0,da,db,dbuf,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dbuf,N*8,hipMemcpyDeviceToHost));
    int pass=1; for(int i=0;i<N&&pass;i++) for(int j=0;j<2;j++){ float e=ha[i][j]*hb[i][j]; if(fabsf(ho[i][j]-e)>1e-4f){fprintf(stderr,"FAIL[%d][%d] %.3f!=%.3f\n",i,j,ho[i][j],e);pass=0;break;}}
    printf("v_pk_mul_f32 packed\n"); printf("%s\n",pass?"PASSED":"FAILED"); return pass?0:1;
}
