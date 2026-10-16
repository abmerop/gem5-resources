/*
 * HIP test for V_CVT_SCALEF32_PK_BF8_F32 (VOP3, narrowing packed scale-convert).
 * gem5 (vop3_cvt.hh Inst_VOP3__V_CVT_SCALE_PK, 32-bit source): src0,src1 = two
 * f32 inputs, src2 = f32 scale.  Two fp8 results are packed into dst[15:0]
 * (out_opsel=0) while dst[31:16] is preserved (the template reads vdst first).
 * With both inputs +0.0 and scale 1.0f the packed fp8 pair is 0 -> dst[15:0]==0
 * and the seeded high half survives.  Base only.
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
    float a=0.0f,b=0.0f,scale=1.0f; uint32_t vd=0xBEEF1234u;
    __asm__ volatile("v_cvt_scalef32_pk_bf8_f32 %0, %1, %2, %3\n\t"
                     :"+v"(vd):"v"(a),"v"(b),"v"(scale));
    out[i]=vd;
}
int main(void){
    uint32_t*ho=(uint32_t*)malloc(N*4),*doo;
    GPU_CHECK(hipMalloc(&doo,N*4));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint16_t lo=ho[i]&0xffff, hi=ho[i]>>16;
        if(lo!=0||hi!=0xBEEF){fprintf(stderr,"FAIL[%d] got %08x (lo %04x hi %04x)\n",i,ho[i],lo,hi);pass=0;break;}}
    if(pass)printf("out[0]=%08x\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
