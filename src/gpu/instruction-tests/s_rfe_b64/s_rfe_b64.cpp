/*
 * HIP test for S_RFE_B64 (ENC_SOP1, return-from-exception) on CDNA4 (gfx950).
 *
 * S_RFE_B64 sets PRIV=0 and PC=S0.u64, returning from a trap/exception
 * handler.  It is architecturally only valid inside a trap handler, so there
 * is no data result to verify directly; the test simply emits the instruction
 * mid-kernel (fed a scalar source pair) after writing a per-lane sentinel and
 * requires the wave to make forward progress.  A correct model must at least
 * decode/execute the opcode without aborting the simulation; if gem5 cannot
 * execute S_RFE_B64 the kernel aborts and no PASSED prints.  Base encoding
 * only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_rfe_b64 s_rfe_b64.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kernel(uint32_t*out,uint64_t rfe_target,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    out[i]=(uint32_t)(i+1);
    /* Emit S_RFE_B64 reading a uniform SGPR pair (kernel arg -> SGPR). */
    __asm__ volatile("s_rfe_b64 %0\n\t"::"s"(rfe_target):"memory");
}
int main(void){
    uint32_t*h_out=(uint32_t*)malloc(N*4);
    uint32_t*d_out; GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemset(d_out,0,N*4));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_out,(uint64_t)0,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++) if(h_out[i]!=(uint32_t)(i+1)){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,h_out[i],i+1);pass=0;break;}
    if(pass)printf("out[0]=%u out[%d]=%u\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
