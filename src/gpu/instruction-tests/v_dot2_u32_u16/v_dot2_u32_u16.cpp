/*
 * HIP test for V_DOT2_U32_U16 (ENC_VOP3P, 2-element unsigned 16-bit dot product
 * accumulated into a u32), pattern A, base only.
 *
 * gem5 (src/arch/amdgpu/common/insts/vop3p.cc):
 *   D.u32 = S0.u16[0]*S1.u16[0] + S0.u16[1]*S1.u16[1] + S2.u32   (no clamp)
 *
 * S2 is an explicit accumulator source (not the dst).  Inputs are chosen small
 * enough that the u32 sum never overflows, so verification is exact.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*a,const uint32_t*b,const uint32_t*c,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t va=a[i],vb=b[i],vc=c[i],d;
    __asm__ volatile("v_dot2_u32_u16 %0, %1, %2, %3\n\t":"=v"(d):"v"(va),"v"(vb),"v"(vc));
    out[i]=d;
}
int main(void){
    uint32_t*h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4),*h_c=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){
        uint32_t a0=(i*7u)&0xffff,a1=(i*13u)&0xffff,b0=(i*3u)&0xffff,b1=(i*5u)&0xffff;
        h_a[i]=(a1<<16)|a0; h_b[i]=(b1<<16)|b0; h_c[i]=(uint32_t)(i*11u);
    }
    uint32_t*d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t a0=h_a[i]&0xffff,a1=h_a[i]>>16,b0=h_b[i]&0xffff,b1=h_b[i]>>16;
        uint32_t e=a0*b0+a1*b1+h_c[i];
        if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,h_out[i],e);pass=0;break;}}
    if(pass)printf("out[1]=%u out[%d]=%u\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
