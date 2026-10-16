/*
 * HIP test for S_BITSET0_B32 (ENC_SOP1, clear one bit of a 32-bit scalar to 0).
 *
 * gem5 (src/arch/amdgpu/common/insts/sop1.cc):
 *   SDST.u32[SSRC0.u[4:0]] = 0
 * gem5 clears the requested bit via replaceBits() on the destination operand
 * without reading the prior destination value, so only the target bit is
 * guaranteed by the model.  This test verifies exactly the defined semantic:
 * after the instruction the bit numbered SSRC0 must read as 0.
 *
 * Pattern B (scalar): bit index in an SGPR ("s"), result via "+s".
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kernel(uint32_t bit, uint32_t seed, uint32_t*out){
    uint32_t sdst=seed;
    __asm__ volatile("s_bitset0_b32 %0, %1\n\t":"+s"(sdst):"s"(bit));
    if(threadIdx.x==0) out[blockIdx.x]=sdst;
}
int main(void){
    uint32_t bits[]={0u,1u,7u,15u,16u,23u,30u,31u};
    const int NC=sizeof(bits)/sizeof(bits[0]);
    uint32_t seed=0xFFFFFFFFu;    /* target bit starts set -> must be cleared */
    uint32_t*d_out; GPU_CHECK(hipMalloc(&d_out,NC*4));
    for(int c=0;c<NC;c++) kernel<<<1,BLOCK_SIZE>>>(bits[c],seed,d_out+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint32_t h_out[64];
    GPU_CHECK(hipMemcpy(h_out,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++) if(((h_out[c]>>bits[c])&1u)!=0u){
        fprintf(stderr,"FAIL[%d]: bit %u not cleared, result=0x%08x\n",c,bits[c],h_out[c]);pass=0;}
    if(pass)printf("bit31 result=0x%08x\n",h_out[NC-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));return pass?0:1;
}
