/*
 * HIP test for V_MADAK_F16 (VOP2 with inline 16-bit literal), pattern A.
 *
 *   D.f16 = S0.f16 * S1.f16 + K      (K is a 16-bit f16 literal in the inst)
 *
 * K is encoded as the f16 constant 2.0 (0x4000).  Verified against
 * s0*s1 + 2.0 computed in float and narrowed to f16, with an f16-scale
 * tolerance.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_madak_f16 v_madak_f16.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
static const float K=2.0f;          /* 0x4000 */
__global__ void kernel(const _Float16*a,const _Float16*b,_Float16*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    _Float16 s0=a[i],s1=b[i],d;
    __asm__ volatile("v_madak_f16 %0, %1, %2, 0x4000\n\t":"=v"(d):"v"(s0),"v"(s1));
    out[i]=d;
}
int main(void){
    _Float16*h_a=(_Float16*)malloc(N*2),*h_b=(_Float16*)malloc(N*2),*h_out=(_Float16*)malloc(N*2);
    for(int i=0;i<N;i++){h_a[i]=(_Float16)((i%13)*0.25f-1.0f);h_b[i]=(_Float16)((i%7)*0.5f+0.5f);}
    _Float16*d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));GPU_CHECK(hipMalloc(&d_out,N*2));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*2,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        float exp=(float)((_Float16)((float)h_a[i]*(float)h_b[i]+K));
        float got=(float)h_out[i];
        if(fabsf(got-exp)>0.02f*fabsf(exp)+0.01f){
            fprintf(stderr,"FAIL[%d] got %g exp %g (a=%g b=%g)\n",i,got,exp,(float)h_a[i],(float)h_b[i]);pass=0;break;}
    }
    if(pass)printf("out[1]=%g out[%d]=%g\n",(float)h_out[1],N-1,(float)h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
