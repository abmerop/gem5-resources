/*
 * HIP test for S_SET_GPR_IDX_MODE (ENC_SOPP, set the GPR-index mode field),
 * pattern B.
 *
 * gem5 (sopp.cc): writes the VGPR-index mode bits (which src/dst operands are
 * relatively addressed) into M0; no program-visible register/memory effect on
 * its own.  The kernel runs a trivial scalar computation with the instruction
 * issued in between (via inline asm) and verifies the surrounding computation.
 * A pass proves S_SET_GPR_IDX_MODE decoded and executed without error.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const int32_t*a,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    int32_t v=a[i];
    __asm__ volatile("s_set_gpr_idx_mode gpr_idx(SRC0,SRC1)\n\t":::"memory","m0");
    out[i]=v*4-9;
}
int main(void){
    int32_t*h_a=(int32_t*)malloc(N*4),*h_out=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_a[i]=i*3+1;
    int32_t*d_a,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){int32_t e=h_a[i]*4-9;
        if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],e);pass=0;break;}}
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
