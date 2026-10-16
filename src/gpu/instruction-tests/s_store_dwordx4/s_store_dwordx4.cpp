/*
 * HIP test for S_STORE_DWORDX4 (ENC_SMEM, scalar store of 4 dwords).  Pattern B.
 *
 * Per gem5 src/arch/amdgpu/common/insts/smem.cc, S_STORE_DWORDX4 writes four
 * dwords (SDATA[0:3]) to memory at SBASE + offset through the scalar memory
 * pipe (plain non-buffer scalar store path).  Each wavefront stores a known
 * 128-bit value to its own 16-byte slot of a global array via the scalar store;
 * the host reads the array back and verifies -- a self-checking round-trip.
 *
 *   s_store_dwordx4 <sdata:4>, <sbase:2>, 0
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_store_dwordx4 s_store_dwordx4.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

typedef unsigned __int128 u128;
static const int NUM_WF=64, BLOCK_SIZE=64;

__global__ void
kernel_s_store_dwordx4(uint32_t base, uint32_t *out)
{
    int wf = blockIdx.x;
    /* Build a distinct 128-bit value per wavefront. */
    u128 val = ((u128)(base + wf) << 96) | ((u128)(base + wf + 1) << 64)
             | ((u128)(base + wf + 2) << 32) | (u128)(base + wf + 3);
    uint32_t *dst = out + wf * 4;
    __asm__ volatile(
        "s_store_dwordx4 %0, %1, 0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :: "s"(val), "s"(dst) : "memory");
}

int
main(void)
{
    const uint32_t base = 0x11110000u;
    uint32_t *h_out=(uint32_t*)malloc(NUM_WF*16);
    for(int i=0;i<NUM_WF*4;i++) h_out[i]=0;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_out,NUM_WF*16));
    GPU_CHECK(hipMemset(d_out,0,NUM_WF*16));
    hipLaunchKernelGGL(kernel_s_store_dwordx4,dim3(NUM_WF),dim3(BLOCK_SIZE),0,0,base,d_out);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,NUM_WF*16,hipMemcpyDeviceToHost));
    int pass=1;
    for(int wf=0;wf<NUM_WF && pass;wf++){
        /* little-endian dword order: dword0 = low 32 bits = base+wf+3, etc. */
        uint32_t e0=base+wf+3, e1=base+wf+2, e2=base+wf+1, e3=base+wf;
        uint32_t g0=h_out[wf*4+0],g1=h_out[wf*4+1],g2=h_out[wf*4+2],g3=h_out[wf*4+3];
        if(g0!=e0||g1!=e1||g2!=e2||g3!=e3){
            fprintf(stderr,"FAIL wf=%d: got %08x %08x %08x %08x exp %08x %08x %08x %08x\n",
                    wf,g0,g1,g2,g3,e0,e1,e2,e3);pass=0;break;}}
    if(pass) printf("out[0..3]=%08x %08x %08x %08x\n",h_out[0],h_out[1],h_out[2],h_out[3]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out)); free(h_out);
    return pass?0:1;
}
