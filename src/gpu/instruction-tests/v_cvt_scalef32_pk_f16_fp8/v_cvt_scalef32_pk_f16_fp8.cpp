/*
 * HIP test for v_cvt_scalef32_pk_f16_fp8 (VOP3 MXFP scalef32 convert, gfx950).
 * gem5 (vop3_cvt.hh Inst_VOP3__V_CVT_SCALE_PK): src0 = packed 2x fp8 data,
 * src1 = f32 scale, dst = 2x f16 (one VGPR dword).  Zero-input trick: data=0,
 * scale=1.0f -> both f16 outputs 0, so the dst dword is 0.  Sentinel + "+v"
 * proves it executed and wrote.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK=64;
__global__ void kern(const uint32_t*in,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t data=in[i]; float scale=1.0f; uint32_t vd=0xDEADBEEFu;
    __asm__ volatile("v_cvt_scalef32_pk_f16_fp8 %0, %1, %2\n\t"
                     : "+v"(vd) : "v"(data), "v"(scale));
    out[i]=vd;
}
int main(void){
    uint32_t*h_in=(uint32_t*)calloc(N,4);
    uint32_t*h_out=(uint32_t*)malloc((size_t)N*4);
    uint32_t*d_in,*d_out;
    GPU_CHECK(hipMalloc(&d_in,N*4)); GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK,BLOCK>>>(d_in,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++) if(h_out[i]!=0u){
        fprintf(stderr,"FAIL out[%d]: got 0x%08x exp 0\n",i,h_out[i]); pass=0; }
    if(pass) printf("all %d f16 dwords==0 (executed)\n",N);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
