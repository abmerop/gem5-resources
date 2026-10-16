/*
 * HIP test for S_SCRATCH_STORE_DWORDX4 (ENC_SMEM, scalar 4-dword scratch
 * store).  Pattern B.
 *
 * gem5 (smem.hh + calcScratchAddr): the final scalar address is
 *   vaddr = SBASE + inst_offset + scaled_offset*64
 * i.e. S_SCRATCH_STORE_DWORDX4 writes the SGPR quad SDATA to the 64-bit
 * address held in SBASE, and S_SCRATCH_LOAD_DWORDX4 reads it back.  We give
 * each wavefront its OWN 16-byte slot (SBASE = scratch + wf*16) so the 64
 * waves never race on the same location, then round-trip store->load and
 * verify all four dwords.  A 128-bit value under the "s" constraint gives an
 * aligned SGPR quad for SDATA.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_scratch_store_dwordx4 \
 *        s_scratch_store_dwordx4.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NUM_WF=64, BLOCK_SIZE=64;   /* one wavefront per block */
__global__ void kernel_s_scratch_store_dwordx4(uint32_t base_val,uint32_t*scratch,uint32_t*out){
    int wf=blockIdx.x;
    uint32_t d0=base_val+(uint32_t)wf;
    uint32_t d1=base_val+(uint32_t)wf+0x1000u;
    uint32_t d2=base_val+(uint32_t)wf+0x2000u;
    uint32_t d3=base_val+(uint32_t)wf+0x3000u;
    unsigned __int128 data=((unsigned __int128)d3<<96)|((unsigned __int128)d2<<64)|
                           ((unsigned __int128)d1<<32)|((unsigned __int128)d0);
    /* per-wavefront private 16-byte slot -> wave-uniform SBASE */
    uint64_t sbase=(uint64_t)(scratch+wf*4);
    unsigned __int128 res;
    __asm__ volatile(
        "s_scratch_store_dwordx4 %1, %2, 0x0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "s_scratch_load_dwordx4  %0, %2, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"=s"(res):"s"(data),"s"(sbase):"memory");
    if(threadIdx.x==0){
        out[wf*4+0]=(uint32_t)(res);
        out[wf*4+1]=(uint32_t)(res>>32);
        out[wf*4+2]=(uint32_t)(res>>64);
        out[wf*4+3]=(uint32_t)(res>>96);
    }
}
int main(void){
    const uint32_t base_val=0xA000;
    uint32_t*h_out=(uint32_t*)malloc(NUM_WF*4*sizeof(uint32_t));
    uint32_t*d_out,*d_scratch;
    GPU_CHECK(hipMalloc(&d_out,NUM_WF*4*sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_scratch,NUM_WF*4*sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_scratch,0,NUM_WF*4*sizeof(uint32_t)));
    hipLaunchKernelGGL(kernel_s_scratch_store_dwordx4,dim3(NUM_WF),dim3(BLOCK_SIZE),0,0,base_val,d_scratch,d_out);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,NUM_WF*4*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int wf=0;wf<NUM_WF;wf++){
        uint32_t e0=base_val+(uint32_t)wf, e1=e0+0x1000u, e2=e0+0x2000u, e3=e0+0x3000u;
        if(h_out[wf*4]!=e0||h_out[wf*4+1]!=e1||h_out[wf*4+2]!=e2||h_out[wf*4+3]!=e3){
            fprintf(stderr,"FAIL wf%d: got %u,%u,%u,%u expected %u,%u,%u,%u\n",wf,
                    h_out[wf*4],h_out[wf*4+1],h_out[wf*4+2],h_out[wf*4+3],e0,e1,e2,e3);pass=0;break;}
    }
    if(pass)printf("out[0]=%u,%u,%u,%u\n",h_out[0],h_out[1],h_out[2],h_out[3]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));GPU_CHECK(hipFree(d_scratch));free(h_out);
    return pass?0:1;
}
