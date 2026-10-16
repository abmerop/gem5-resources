/*
 * HIP test for S_ATC_PROBE (ENC_SMEM, address-translation-cache probe hint).
 * Pattern B.
 *
 * S_ATC_PROBE is a hint that probes the ATC/TLB for the line at
 * (SBASE + OFFSET); it produces no architectural result.  This test issues the
 * probe against a valid buffer base and then writes a sentinel so that, if the
 * instruction executes without faulting, the self-check passes.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_atc_probe s_atc_probe.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kernel(uint32_t*base,uint32_t*out){
    uint64_t addr=(uint64_t)base;
    __asm__ volatile(
        "s_atc_probe 0, %0, 0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        ::"s"(addr):"memory");
    if(threadIdx.x==0) out[0]=0xA5A5A5A5u;
}
int main(void){
    uint32_t h_out=0;
    uint32_t*d_base,*d_out;
    GPU_CHECK(hipMalloc(&d_base,64));GPU_CHECK(hipMalloc(&d_out,4));
    GPU_CHECK(hipMemset(d_base,0,64));GPU_CHECK(hipMemset(d_out,0,4));
    kernel<<<1,BLOCK_SIZE>>>(d_base,d_out);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(&h_out,d_out,4,hipMemcpyDeviceToHost));
    int pass=(h_out==0xA5A5A5A5u);
    if(!pass)fprintf(stderr,"FAIL out=0x%x (exp 0xA5A5A5A5)\n",h_out);
    else printf("out=0x%x\n",h_out);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
