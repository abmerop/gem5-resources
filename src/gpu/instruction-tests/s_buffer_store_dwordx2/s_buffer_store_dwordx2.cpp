/*
 * HIP test for S_BUFFER_STORE_DWORDX2 (ENC_SMEM) on CDNA4 (gfx950).  Pattern B.
 *
 *   MEM[ base + offset ] = SDATA[0:1]   (scalar store of 2 dwords through a
 *   128-bit buffer resource descriptor)
 *
 * A single wavefront writes 2 dwords held in SGPRs to a global array via the
 * buffer resource descriptor; the host verifies the round-trip.  Base encoding
 * only (SMEM has no DPP/SDWA).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_buffer_store_dwordx2 s_buffer_store_dwordx2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32xW __attribute__((ext_vector_type(2)));
static const int BLOCK_SIZE=64, W=2;

__global__ void kernel(uint32_t*mem,uint32_t v0,uint32_t v1){
    uint64_t base=(uint64_t)mem;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    srd.w=0x00020000u;
    u32xW d; d[0]=v0; d[1]=v1;
    __asm__ volatile(
        "s_buffer_store_dwordx2 %0, %1, 0\n\t"
        "s_dcache_wb\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        ::"s"(d),"s"(srd):"memory");
}

int main(void){
    uint32_t h_mem[2]={0,0}, h_out[2];
    uint32_t v0=0xC0DE0000u, v1=0xC0DE1111u;
    uint32_t*d_mem;
    GPU_CHECK(hipMalloc(&d_mem,W*4));
    GPU_CHECK(hipMemcpy(d_mem,h_mem,W*4,hipMemcpyHostToDevice));
    kernel<<<1,BLOCK_SIZE>>>(d_mem,v0,v1);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_mem,W*4,hipMemcpyDeviceToHost));
    int pass=1;
    uint32_t exp[2]={v0,v1};
    for(int i=0;i<W;i++) if(h_out[i]!=exp[i]){
        fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],exp[i]);pass=0;break;}
    if(pass) printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],W-1,h_out[W-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_mem));
    return pass?0:1;
}
