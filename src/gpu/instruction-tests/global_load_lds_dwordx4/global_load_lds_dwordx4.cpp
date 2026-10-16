/*
 * HIP test for GLOBAL_LOAD_LDS_DWORDX4 (ENC_FLAT_GLBL, direct global->LDS DMA
 * load of 4 dwords per lane).
 *
 * This is an LDS-DMA load: each active lane reads 16 bytes (4 dwords) from its
 * global address and the hardware writes them into LDS.  Per the gem5 model
 * (ldsComplete<4> in vega/insts/op_encodings.hh) lane L writes its 4 dwords to
 *   lds_addr + L*4*sizeof(dword) + i*sizeof(dword),  i in [0,4)
 * where lds_addr = m0[17:2]*4 + inst_offset.  So lane L's dwords land in
 * lds[L*4 + 0..3] relative to the m0 LDS base.
 *
 * The gfx950 assembler only accepts the SADDR form for this op, so the global
 * base address is supplied via an SGPR pair (saddr) and the per-lane byte
 * offset via a VGPR (vaddr).  There is no VGPR destination; m0 supplies the LDS
 * base offset.  Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void
kernel(const uint32_t *g, uint32_t *out, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    __shared__ uint32_t lds[BLOCK_SIZE*4];   /* 4 dwords per lane */
    int t = threadIdx.x;

    /* Per-wavefront global base (4 dwords per lane). */
    const uint32_t *base = g + (size_t)blockIdx.x * BLOCK_SIZE * 4;
    uint64_t saddr = (uint64_t)(uintptr_t)base;
    uint32_t voff  = (uint32_t)(t * 16);           /* byte offset, 16B/lane */
    uint32_t lds_off = (uint32_t)(uintptr_t)lds;   /* LDS base offset */

    __asm__ volatile(
        "s_mov_b32 m0, %2\n\t"
        "global_load_lds_dwordx4 %0, %1\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        :
        : "v"(voff), "s"(saddr), "s"(lds_off)
        : "memory", "m0");

    __syncthreads();
    for (int j=0;j<4;j++)
        out[i*4+j] = lds[t*4+j];
}

int main(void)
{
    uint32_t *h_g=(uint32_t*)malloc((size_t)N*4*4),*h_out=(uint32_t*)malloc((size_t)N*4*4);
    for(size_t k=0;k<(size_t)N*4;k++) h_g[k]=(uint32_t)(k*0x9e3779b9u+11u);
    uint32_t *d_g,*d_out;
    GPU_CHECK(hipMalloc(&d_g,(size_t)N*4*4));GPU_CHECK(hipMalloc(&d_out,(size_t)N*4*4));
    GPU_CHECK(hipMemcpy(d_g,h_g,(size_t)N*4*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_g,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,(size_t)N*4*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(size_t k=0;k<(size_t)N*4;k++){
        if(h_out[k]!=h_g[k]){fprintf(stderr,"FAIL[%zu] got 0x%08x exp 0x%08x\n",k,h_out[k],h_g[k]);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],N*4-1,h_out[N*4-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
