/*
 * HIP test for V_CVT_SCALEF32_SR_PK_FP4_F32 (VOP3, SR packed fp4 convert).
 * gem5 (vop3_cvt.hh Inst_VOP3__V_CVT_SCALE_SR_PK_FP4, 32-bit source): src0 is
 * a pair of f32 inputs, src1=i32 seed, src2=f32 scale; packs two fp4 results
 * into byte0 of vdst (OPSEL[3:2]=0), preserving the rest.  With both inputs
 * +0.0 the fp4 pair is 0 regardless of seed -> byte0 clears.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef float f32x2 __attribute__((ext_vector_type(2)));
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    f32x2 data; data[0]=0.0f; data[1]=0.0f;
    int32_t seed=(int32_t)(0x9E3779B9u*(uint32_t)(i+1)); float scale=1.0f;
    uint32_t d=0xFFFFFFFFu;
    __asm__ volatile("v_cvt_scalef32_sr_pk_fp4_f32 %0, %1, %2, %3\n\t"
        :"+v"(d):"v"(data),"v"(seed),"v"(scale));
    out[i]=d;
}
int main(void){
    uint32_t*ho=(uint32_t*)malloc(N*4),*doo; GPU_CHECK(hipMalloc(&doo,N*4));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint32_t b0=ho[i]&0xff; if(b0!=0){fprintf(stderr,"FAIL[%d] byte0=%02x full %08x\n",i,b0,ho[i]);pass=0;break;}}
    if(pass)printf("out[0]=%08x byte0=%02x\n",ho[0],ho[0]&0xff);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
