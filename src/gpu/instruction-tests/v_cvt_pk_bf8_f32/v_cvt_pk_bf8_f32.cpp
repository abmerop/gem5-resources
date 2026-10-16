/*
 * HIP test for V_CVT_PK_BF8_F32 (VOP3, pack two f32 -> two bf8 in the low 16
 * bits of the dst, preserving the high 16 bits).
 * gem5 (vop3.cc): reads vdst (preserve), writes packed bf8 pair into bits[15:0].
 * Zero-input trick: both f32 sources = 0.0 -> both bf8 = 0x00 -> low 16 bits 0.
 * dst seeded with a sentinel and read-modified ("+v"); we verify low16==0 and
 * the preserved high 16 bits.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
#define SENT 0xBEEF0000u
__global__ void kern(const float*a,const float*b,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i], s1=b[i]; uint32_t d=SENT;
    __asm__ volatile("v_cvt_pk_bf8_f32 %0, %1, %2\n\t" : "+v"(d) : "v"(s0), "v"(s1));
    out[i]=d;
}
int main(void){
    float*h_a=(float*)calloc(N,4),*h_b=(float*)calloc(N,4);
    uint32_t*h_out=(uint32_t*)malloc(N*4);
    float*d_a,*d_b; uint32_t*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint32_t exp=(SENT & 0xffff0000u);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x (low16 packed bf8 = 0)\n",h_out[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
