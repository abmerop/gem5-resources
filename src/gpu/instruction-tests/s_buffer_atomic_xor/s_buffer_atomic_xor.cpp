/*
 * HIP test for S_BUFFER_ATOMIC_XOR (ENC_SMEM, 32-bit scalar buffer atomic
 * xor, return-old).  Pattern B.
 *
 *   tmp = MEM[addr]
 *   MEM[addr] = tmp ^ SDATA              (32-bit xor)
 *   if (GLC) SDATA = tmp                 (old value)
 *
 * Addresses memory through a 128-bit buffer resource descriptor (4 SGPRs).
 * SDATA is in/out ("+s"); one wavefront performs a single uniform atomic.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_buffer_atomic_xor \
 *        s_buffer_atomic_xor.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int BLOCK_SIZE=64;
__global__ void kernel(uint32_t*mem,uint32_t dataval,uint32_t*old_out){
    uint64_t base=(uint64_t)mem;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    srd.w=0x00020000u;
    uint32_t d=dataval;
    __asm__ volatile(
        "s_buffer_atomic_xor %0, %1, 0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"+s"(d):"s"(srd):"memory");
    if(threadIdx.x==0) old_out[0]=d;
}
int main(void){
    const uint32_t init=0xA5A5A5A5u, dataval=0x0F0F0F0Fu; /* xor -> 0xAAAAAAAA */
    uint32_t h_mem=0,h_old=0;
    uint32_t*d_mem,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,4));GPU_CHECK(hipMalloc(&d_old,4));
    GPU_CHECK(hipMemcpy(d_mem,&init,4,hipMemcpyHostToDevice));
    kernel<<<1,BLOCK_SIZE>>>(d_mem,dataval,d_old);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(&h_mem,d_mem,4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(&h_old,d_old,4,hipMemcpyDeviceToHost));
    uint32_t exp_old=init, exp_mem=init^dataval;
    int pass=(h_old==exp_old)&&(h_mem==exp_mem);
    if(!pass)fprintf(stderr,"FAIL old=0x%08x(exp 0x%08x) mem=0x%08x(exp 0x%08x)\n",h_old,exp_old,h_mem,exp_mem);
    else printf("old=0x%08x mem=0x%08x\n",h_old,h_mem);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
