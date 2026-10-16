/*
 * HIP test for S_BUFFER_LOAD_DWORDX2 (ENC_SMEM, scalar load of two dwords from
 * a buffer resource into an SGPR pair).  Pattern B.
 *
 *   SDATA[0:1] = MEM[ base + offset : +8 ]
 *
 * Addresses memory through a 128-bit buffer resource descriptor (4 SGPRs),
 * same scaffolding as S_BUFFER_LOAD_DWORD.  Loads one 64-bit value.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_buffer_load_dwordx2 \
 *        s_buffer_load_dwordx2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int BLOCK_SIZE=64;

__global__ void kernel(uint32_t*mem,uint64_t*out){
    uint64_t base=(uint64_t)mem;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    srd.w=0x00020000u;
    uint64_t d;
    __asm__ volatile(
        "s_buffer_load_dwordx2 %0, %1, 0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"=s"(d):"s"(srd):"memory");
    if(threadIdx.x==0) out[0]=d;
}

int main(void){
    uint32_t h_mem[2]={0xdeadbeefu, 0x12345678u};
    uint64_t h_out=0;
    uint32_t*d_mem; uint64_t*d_out;
    GPU_CHECK(hipMalloc(&d_mem,8));GPU_CHECK(hipMalloc(&d_out,8));
    GPU_CHECK(hipMemcpy(d_mem,h_mem,8,hipMemcpyHostToDevice));
    kernel<<<1,BLOCK_SIZE>>>(d_mem,d_out);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(&h_out,d_out,8,hipMemcpyDeviceToHost));
    uint64_t exp=((uint64_t)h_mem[1]<<32)|(uint64_t)h_mem[0];
    int pass=(h_out==exp);
    if(!pass)fprintf(stderr,"FAIL d=0x%016llx(exp 0x%016llx)\n",
                     (unsigned long long)h_out,(unsigned long long)exp);
    else printf("d=0x%016llx\n",(unsigned long long)h_out);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
