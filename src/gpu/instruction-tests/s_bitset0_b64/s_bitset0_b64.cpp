/*
 * HIP test for S_BITSET0_B64 (SOP1 scalar clear-bit).
 *
 * gem5 (sop1.cc): sdst.setBit(bits(SSRC0, 5, 0), 0) -- it clears bit
 * (SSRC0 & 0x3f) of the 64-bit SDST to 0.  Note gem5 does NOT read SDST first,
 * so only the *target* bit is architecturally deterministic; the test verifies
 * exactly that: after S_BITSET0_B64 with index k, bit k of the result is 0.
 * The bit index is a uniform kernel arg (scalar).  Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

__device__ __forceinline__ uint64_t bitset0(uint32_t idx){
    uint64_t d;
    __asm__ volatile("s_bitset0_b64 %0, %1\n\t":"=s"(d):"s"(idx));
    return d;
}
__global__ void
kern(uint32_t k0,uint32_t k1,uint32_t k2,uint32_t k3,uint32_t k4,uint32_t k5,
     uint64_t*out){
    if(threadIdx.x!=0) return;
    out[0]=bitset0(k0); out[1]=bitset0(k1); out[2]=bitset0(k2);
    out[3]=bitset0(k3); out[4]=bitset0(k4); out[5]=bitset0(k5);
}
int main(void){
    uint32_t k[6]={0,7,31,32,45,63};
    uint64_t *d_o, ho[6];
    GPU_CHECK(hipMalloc(&d_o,6*8));
    kern<<<1,64>>>(k[0],k[1],k[2],k[3],k[4],k[5],d_o);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,d_o,6*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<6;i++){ if((ho[i]>>k[i])&1ULL){
        fprintf(stderr,"FAIL k=%u result bit set: %016llx\n",k[i],(unsigned long long)ho[i]);pass=0;} }
    if(pass)printf("cleared bits 0,7,31,32,45,63 all 0\n");
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
