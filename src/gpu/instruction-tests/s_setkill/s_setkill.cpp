/*
 * HIP test for S_SETKILL (ENC_SOPP, wave-control).  Pattern B, base only.
 *
 * S_SETKILL sets the wavefront KILL bit to SIMM16[0].  Using SIMM16=0 is a
 * no-op with respect to the wave (it does not kill it), which lets a normal
 * self-checking kernel run to completion while still exercising the opcode.
 * The compiler never emits S_SETKILL from HIP, so it is placed via inline asm
 * with a "memory" clobber so it cannot be optimized away.  The surrounding
 * kernel computes out[i] = in[i] + 1 so the host can confirm the wave really
 * executed past the instruction.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   s_setkill 0
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o s_setkill s_setkill.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kernel(const uint32_t*in,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t v=in[i];
    /* S_SETKILL 0 : no-op kill bit; must survive past it. */
    __asm__ volatile("s_setkill 0\n\t":::"memory");
    out[i]=v+1u;
}
int main(void){
    uint32_t*h_in=(uint32_t*)malloc(N*4);
    uint32_t*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_in[i]=(uint32_t)(i*2654435761u);
    uint32_t*d_in,*d_out;
    GPU_CHECK(hipMalloc(&d_in,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t exp=h_in[i]+1u;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%x exp 0x%x\n",
            i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%x out[%d]=0x%x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
