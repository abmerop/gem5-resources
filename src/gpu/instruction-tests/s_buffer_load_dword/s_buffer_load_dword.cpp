/*
 * HIP test for S_BUFFER_LOAD_DWORD (ENC_SMEM, scalar load of one dword from a
 * buffer resource).  Pattern B.
 *
 *   SDATA = MEM[ base + offset ]
 *
 * Addresses memory through a 128-bit buffer resource descriptor (4 SGPRs),
 * same scaffolding as the s_buffer_atomic tests.  Loads two dwords (offset 0
 * and 4) and checks both.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_buffer_load_dword s_buffer_load_dword.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int BLOCK_SIZE=64;

__global__ void kernel(uint32_t*mem,uint32_t*out){
    uint64_t base=(uint64_t)mem;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    srd.w=0x00020000u;
    uint32_t d0, d1;
    __asm__ volatile(
        "s_buffer_load_dword %0, %2, 0\n\t"
        "s_buffer_load_dword %1, %2, 4\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"=s"(d0),"=s"(d1):"s"(srd):"memory");
    if(threadIdx.x==0){ out[0]=d0; out[1]=d1; }
}

int main(void){
    uint32_t h_mem[2]={0xdeadbeefu, 0x12345678u};
    uint32_t h_out[2]={0,0};
    uint32_t*d_mem,*d_out;
    GPU_CHECK(hipMalloc(&d_mem,8));GPU_CHECK(hipMalloc(&d_out,8));
    GPU_CHECK(hipMemcpy(d_mem,h_mem,8,hipMemcpyHostToDevice));
    kernel<<<1,BLOCK_SIZE>>>(d_mem,d_out);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,8,hipMemcpyDeviceToHost));
    int pass=(h_out[0]==h_mem[0])&&(h_out[1]==h_mem[1]);
    if(!pass)fprintf(stderr,"FAIL d0=0x%08x(exp 0x%08x) d1=0x%08x(exp 0x%08x)\n",
                     h_out[0],h_mem[0],h_out[1],h_mem[1]);
    else printf("d0=0x%08x d1=0x%08x\n",h_out[0],h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
