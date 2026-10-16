/*
 * HIP test for V_CVT_SCALEF32_F16_BF8 (VOP3, widening scale-convert).
 * gem5 (vop3_cvt.hh Inst_VOP3__V_CVT_SCALE): src0=packed byte data, src1=f32
 * scale; converts the opsel-selected fp8 byte to the dst float, scaled, and
 * clobbers the whole dst.  With data=0 (fp8 +0.0) and scale=1.0f the output is
 * exactly +0.0 -> dst reads back 0, proving the datapath wrote.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t data=0u; float scale=1.0f; uint32_t vd=0xDEADBEEFu;
    __asm__ volatile("v_cvt_scalef32_f16_bf8 %0, %1, %2\n\t":"+v"(vd):"v"(data),"v"(scale));
    out[i]=vd;
}
int main(void){
    uint32_t*ho=(uint32_t*)malloc(N*4),*doo;
    GPU_CHECK(hipMalloc(&doo,N*4));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ if(ho[i]!=0u){fprintf(stderr,"FAIL[%d] got %08x exp 0\n",i,ho[i]);pass=0;break;}}
    if(pass)printf("out[0]=%08x\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
