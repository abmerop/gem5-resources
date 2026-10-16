/*
 * HIP test for S_SCRATCH_STORE_DWORDX2 (ENC_SMEM, scalar 2-dword scratch
 * store).  Pattern B.
 *
 * gem5 (smem.hh + calcScratchAddr): the final scalar address is
 *   vaddr = SBASE + inst_offset + scaled_offset*64
 * i.e. S_SCRATCH_STORE_DWORDX2 writes the SGPR pair SDATA to the 64-bit
 * address held in SBASE, and S_SCRATCH_LOAD_DWORDX2 reads it back.  We give
 * each wavefront its OWN 8-byte slot (SBASE = scratch + wf*8) so the 64 waves
 * never race on the same location, then round-trip store->load and verify.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_scratch_store_dwordx2 \
 *        s_scratch_store_dwordx2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NUM_WF=64, BLOCK_SIZE=64;   /* one wavefront per block */
__global__ void kernel_s_scratch_store_dwordx2(uint32_t base_val,uint32_t*scratch,uint32_t*out){
    int wf=blockIdx.x;
    uint32_t d0=base_val+(uint32_t)wf;
    uint32_t d1=base_val+(uint32_t)wf+0x1000u;
    uint64_t data=((uint64_t)d1<<32)|(uint64_t)d0;
    /* per-wavefront private 8-byte slot -> wave-uniform SBASE */
    uint64_t sbase=(uint64_t)(scratch+wf*2);
    uint64_t res;
    __asm__ volatile(
        "s_scratch_store_dwordx2 %1, %2, 0x0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "s_scratch_load_dwordx2  %0, %2, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"=s"(res):"s"(data),"s"(sbase):"memory");
    if(threadIdx.x==0){ out[wf*2+0]=(uint32_t)(res&0xffffffffu); out[wf*2+1]=(uint32_t)(res>>32); }
}
int main(void){
    const uint32_t base_val=0xA000;
    uint32_t*h_out=(uint32_t*)malloc(NUM_WF*2*sizeof(uint32_t));
    uint32_t*d_out,*d_scratch;
    GPU_CHECK(hipMalloc(&d_out,NUM_WF*2*sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_scratch,NUM_WF*2*sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_scratch,0,NUM_WF*2*sizeof(uint32_t)));
    hipLaunchKernelGGL(kernel_s_scratch_store_dwordx2,dim3(NUM_WF),dim3(BLOCK_SIZE),0,0,base_val,d_scratch,d_out);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,NUM_WF*2*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int wf=0;wf<NUM_WF;wf++){
        uint32_t e0=base_val+(uint32_t)wf, e1=base_val+(uint32_t)wf+0x1000u;
        if(h_out[wf*2]!=e0||h_out[wf*2+1]!=e1){
            fprintf(stderr,"FAIL wf%d: got %u,%u expected %u,%u\n",wf,h_out[wf*2],h_out[wf*2+1],e0,e1);pass=0;break;}
    }
    if(pass)printf("out[0]=%u,%u\n",h_out[0],h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));GPU_CHECK(hipFree(d_scratch));free(h_out);
    return pass?0:1;
}
