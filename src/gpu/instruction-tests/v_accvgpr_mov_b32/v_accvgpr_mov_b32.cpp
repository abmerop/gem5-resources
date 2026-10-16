/*
 * HIP test for V_ACCVGPR_MOV_B32 (VOP1, AGPR-to-AGPR 32-bit move), pattern A.
 *
 *   A_dst.b32 = A_src.b32
 *
 * The ISA XML lists DPP/SDWA variants for this opcode, but the gfx950
 * assembler (llvm-mc -mcpu=gfx950) rejects both ("dpp/sdwa variant of this
 * instruction is not supported"), so only the base form can be authored;
 * variants exercised = base.
 *
 * The value is forced into an AGPR ("a" constraint), moved AGPR->AGPR with
 * v_accvgpr_mov_b32, then read back.  Verified as an identity copy.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_accvgpr_mov_b32 v_accvgpr_mov_b32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const int32_t*a,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    int32_t v=a[i],d;
    __asm__ volatile("v_accvgpr_mov_b32 %0, %1\n\t":"=a"(d):"a"(v));
    out[i]=d;
}
int main(void){
    int32_t*h_a=(int32_t*)malloc(N*4),*h_out=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_a[i]=(int32_t)(i*2654435761u+12345u);
    int32_t*d_a,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++) if(h_out[i]!=h_a[i]){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],h_a[i]);pass=0;break;}
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
