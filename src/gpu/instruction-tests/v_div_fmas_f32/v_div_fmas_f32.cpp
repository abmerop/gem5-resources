/*
 * HIP test for V_DIV_FMAS_F32 (VOP3): vdst = VCC ? 2^32*fma(a,b,c) : fma(a,b,c)
 * (gem5 vop3.cc reads VCC).  We force VCC=0 for the whole wave (s_mov_b64 vcc,0)
 * so the result is the plain fused-multiply-add fma(a,b,c).  Small exact integer
 * operands -> bit-exact against fmaf.  Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const float*a,const float*b,const float*c,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i],s1=b[i],s2=c[i],r;
    __asm__ volatile("s_mov_b64 vcc, 0\n\tv_div_fmas_f32 %0, %1, %2, %3\n\t"
                     :"=v"(r):"v"(s0),"v"(s1),"v"(s2):"vcc");
    out[i]=r;
}
int main(void){
    float*ha=(float*)malloc(N*4),*hb=(float*)malloc(N*4),*hc=(float*)malloc(N*4),*ho=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ha[i]=(float)(i%7)+1;hb[i]=(float)(i%5)+2;hc[i]=(float)(i%11);}
    float*da,*db,*dc,*dO;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){float e=fmaf(ha[i],hb[i],hc[i]);
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,ho[i],e);pass=0;break;}}
    if(pass)printf("out[0]=%g\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
