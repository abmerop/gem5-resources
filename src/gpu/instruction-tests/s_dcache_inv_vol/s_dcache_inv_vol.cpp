/*
 * HIP test for S_DCACHE_INV_VOL (ENC_SMEM, invalidate volatile lines in the
 * scalar data cache).  Pattern B.
 *
 * S_DCACHE_INV_VOL produces no architectural result; it is a cache-maintenance
 * hint.  This test issues the invalidate and then does a normal global store so
 * that, if the instruction executes without faulting, the self-check passes.
 * (If gem5 lacks the instruction the kernel aborts and no PASSED prints.)
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   s_dcache_inv_vol
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_dcache_inv_vol \
 *        s_dcache_inv_vol.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kernel(uint32_t*out){
    __asm__ volatile("s_dcache_inv_vol\n\t"
                     "s_waitcnt lgkmcnt(0)\n\t":::"memory");
    if(threadIdx.x==0) out[0]=0xA5A5A5A5u;
}
int main(void){
    uint32_t h_out=0;
    uint32_t*d_out;
    GPU_CHECK(hipMalloc(&d_out,4));
    GPU_CHECK(hipMemset(d_out,0,4));
    kernel<<<1,BLOCK_SIZE>>>(d_out);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(&h_out,d_out,4,hipMemcpyDeviceToHost));
    int pass=(h_out==0xA5A5A5A5u);
    if(!pass)fprintf(stderr,"FAIL out=0x%x (exp 0xA5A5A5A5)\n",h_out);
    else printf("out=0x%x\n",h_out);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
