/*
 * HIP test for S_BUFFER_STORE_DWORDX4 (ENC_SMEM) on CDNA4 (gfx950).  Pattern B.
 *
 *   MEM[ base + offset ] = SDATA[0:3]   (scalar store of 4 dwords through a
 *   128-bit buffer resource descriptor)
 *
 * A single wavefront writes 4 dwords held in SGPRs to a global array via the
 * buffer resource descriptor; the host verifies the round-trip.  Base encoding
 * only (SMEM has no DPP/SDWA).
 *
 * gem5 status note: Inst_SMEM__S_BUFFER_STORE_DWORDX4::execute() is
 * panicUnimplemented() in src/arch/amdgpu/common/insts/smem.cc (the store half
 * of the S_BUFFER_* family is still stubbed; loads/atomics were fixed by the
 * SBASE<<1 change).  This test is expected to abort the simulator.  The
 * verification below is exact for a correct implementation.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_buffer_store_dwordx4 s_buffer_store_dwordx4.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int BLOCK_SIZE=64, W=4;

__global__ void kernel(uint32_t*mem,uint32_t v0,uint32_t v1,uint32_t v2,uint32_t v3){
    uint64_t base=(uint64_t)mem;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    srd.w=0x00020000u;
    u32x4 d; d[0]=v0; d[1]=v1; d[2]=v2; d[3]=v3;
    __asm__ volatile(
        "s_buffer_store_dwordx4 %0, %1, 0\n\t"
        "s_dcache_wb\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        ::"s"(d),"s"(srd):"memory");
}

int main(void){
    uint32_t h_mem[4]={0,0,0,0}, h_out[4];
    uint32_t v[4]={0xC0DE0000u,0xC0DE1111u,0xC0DE2222u,0xC0DE3333u};
    uint32_t*d_mem;
    GPU_CHECK(hipMalloc(&d_mem,W*4));
    GPU_CHECK(hipMemcpy(d_mem,h_mem,W*4,hipMemcpyHostToDevice));
    kernel<<<1,BLOCK_SIZE>>>(d_mem,v[0],v[1],v[2],v[3]);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_mem,W*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<W;i++) if(h_out[i]!=v[i]){
        fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],v[i]);pass=0;break;}
    if(pass) printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],W-1,h_out[W-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_mem));
    return pass?0:1;
}
