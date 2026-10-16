/*
 * HIP test for v_cvt_scalef32_pk_f32_fp8 (VOP3 MXFP scalef32 convert, gfx950).
 * gem5 (vop3_cvt.hh Inst_VOP3__V_CVT_SCALE_PK): src0 = packed 2x fp8 data,
 * src1 = f32 scale, dst = 2x f32 (VGPR pair).  Zero-input trick: data = 0
 * (both fp8 components +0.0), scale = 1.0f -> both f32 outputs = 0.  The dst
 * pair is seeded with a sentinel and read-modified ("+v") so a pass proves the
 * instruction executed and wrote the computed (all-zero) result.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u2 __attribute__((ext_vector_type(2)));
static const int N=256, BLOCK=64;
__global__ void kern(const uint32_t*in,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t data=in[i];            /* 0 -> two fp8 +0.0 */
    float scale=1.0f;
    u2 vd; vd[0]=0xDEADBEEFu; vd[1]=0xDEADBEEFu;
    __asm__ volatile("v_cvt_scalef32_pk_f32_fp8 %0, %1, %2\n\t"
                     : "+v"(vd) : "v"(data), "v"(scale));
    out[i*2]=vd[0]; out[i*2+1]=vd[1];
}
int main(void){
    uint32_t*h_in=(uint32_t*)calloc(N,4);
    uint32_t*h_out=(uint32_t*)malloc((size_t)N*2*4);
    uint32_t*d_in,*d_out;
    GPU_CHECK(hipMalloc(&d_in,N*4)); GPU_CHECK(hipMalloc(&d_out,(size_t)N*2*4));
    GPU_CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK,BLOCK>>>(d_in,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,(size_t)N*2*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N*2&&pass;i++) if(h_out[i]!=0u){
        fprintf(stderr,"FAIL out[%d]: got 0x%08x exp 0\n",i,h_out[i]); pass=0; }
    if(pass) printf("all %d f32 dwords==0 (executed)\n",N*2);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
