/*
 * HIP test for v_cvt_scalef32_pk32_f32_bf6 (VOP3 MXFP scalef32 pack-32 convert, gfx950/CDNA4).
 * Zero-input trick: all 6 input dwords = 0 (all components +0.0),
 * scale = 1.0 -> every converted output component = 0, so all 32
 * output dwords must be exactly 0. The 32 dst VGPRs are seeded with a
 * nonzero sentinel and read-modified ("+v"), so a pass proves the instruction
 * executed and wrote the computed (all-zero) result.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);}} while(0)
typedef uint32_t uin __attribute__((ext_vector_type(6)));
typedef uint32_t uout __attribute__((ext_vector_type(32)));
static const int N=256, BLOCK=64, OUT=32, IN=6;
__global__ void kern(const uint32_t*in,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    uin vs;
    for(int k=0;k<IN;k++) vs[k]=in[i*IN+k];
    float scale=1.0f;
    uout vd;
    for(int k=0;k<OUT;k++) vd[k]=0xDEADBEEFu;
    __asm__ volatile("v_cvt_scalef32_pk32_f32_bf6 %0, %1, %2\n\t" : "+v"(vd) : "v"(vs),"v"(scale));
    for(int k=0;k<OUT;k++) out[i*OUT+k]=vd[k];
}
int main(void){
    uint32_t*h_in=(uint32_t*)calloc((size_t)N*IN,4);
    uint32_t*h_out=(uint32_t*)malloc((size_t)N*OUT*4);
    uint32_t*d_in,*d_out;
    GPU_CHECK(hipMalloc(&d_in,(size_t)N*IN*4));
    GPU_CHECK(hipMalloc(&d_out,(size_t)N*OUT*4));
    GPU_CHECK(hipMemcpy(d_in,h_in,(size_t)N*IN*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kern,dim3(N/BLOCK),dim3(BLOCK),0,0,d_in,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,(size_t)N*OUT*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N*OUT&&pass;i++) if(h_out[i]!=0u){
        fprintf(stderr,"FAIL out[%d]: got 0x%08x exp 0\n",i,h_out[i]); pass=0; }
    if(pass) printf("all %d dwords==0 (executed)\n",N*OUT);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
