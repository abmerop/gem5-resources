/*
 * HIP test for S_STORE_DWORDX2 (ENC_SMEM, scalar store of 2 dwords).  Pattern B.
 *
 * Per gem5 src/arch/amdgpu/common/insts/smem.cc, S_STORE_DWORDX2 writes two
 * dwords (SDATA[0:1]) to memory at SBASE + offset through the scalar memory
 * pipe (plain non-buffer scalar store path).  Each wavefront stores a known
 * 64-bit value to its own slot of a global array via the scalar store; the host
 * reads the array back and verifies -- a self-checking round-trip.
 *
 *   s_store_dwordx2 <sdata:2>, <sbase:2>, 0
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_store_dwordx2 s_store_dwordx2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

static const int NUM_WF=64, BLOCK_SIZE=64;

__global__ void
kernel_s_store_dwordx2(uint64_t base, uint64_t *out)
{
    int wf = blockIdx.x;
    uint64_t val = base + (uint64_t)wf;
    uint64_t *dst = out + wf;
    __asm__ volatile(
        "s_store_dwordx2 %0, %1, 0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :: "s"(val), "s"(dst) : "memory");
}

int
main(void)
{
    const uint64_t base = 0xABCDEF0000000000ULL;
    uint64_t *h_out=(uint64_t*)malloc(NUM_WF*8);
    for(int i=0;i<NUM_WF;i++) h_out[i]=0;
    uint64_t *d_out;
    GPU_CHECK(hipMalloc(&d_out,NUM_WF*8));
    GPU_CHECK(hipMemset(d_out,0,NUM_WF*8));
    hipLaunchKernelGGL(kernel_s_store_dwordx2,dim3(NUM_WF),dim3(BLOCK_SIZE),0,0,base,d_out);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,NUM_WF*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int wf=0;wf<NUM_WF;wf++){ uint64_t e=base+(uint64_t)wf;
        if(h_out[wf]!=e){fprintf(stderr,"FAIL out[%d]: got 0x%016llx expected 0x%016llx\n",wf,(unsigned long long)h_out[wf],(unsigned long long)e);pass=0;break;}}
    if(pass) printf("out[0]=0x%016llx out[%d]=0x%016llx\n",(unsigned long long)h_out[0],NUM_WF-1,(unsigned long long)h_out[NUM_WF-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out)); free(h_out);
    return pass?0:1;
}
