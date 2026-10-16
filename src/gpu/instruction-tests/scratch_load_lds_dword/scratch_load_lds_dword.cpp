/*
 * HIP test for SCRATCH_LOAD_LDS_DWORD (ENC_FLAT scratch segment, direct
 * scratch/private -> LDS DMA load of a 32-bit dword).  Shares the
 * Inst_FLAT__FLAT_LOAD_LDS_DWORD implementation in gem5.  No VGPR destination.
 *
 * Per-lane round trip entirely in gem5's memories (same idiom as
 * scratch_load_lds_sbyte): scratch_store_dword writes a known value into this
 * lane's private scratch slot, scratch_load_lds_dword DMAs it into lds[lane],
 * then out[i] = lds[threadIdx.x].  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const uint32_t*g,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    volatile int pad[8]; pad[t&7]=i;              /* force scratch segment */
    uint32_t val=g[i];
    uint32_t lds_off=(uint32_t)(uintptr_t)lds;
    __asm__ volatile(
        "scratch_store_dword off, %1, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "s_mov_b32 m0, %2\n\t"
        "scratch_load_lds_dword off, off offset:0\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        : : "v"(lds), "v"(val), "s"(lds_off) : "memory","m0");
    __syncthreads();
    out[i]=lds[t];
}
int main(void){
    uint32_t*h_g=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_g[i]=(uint32_t)(i*2654435761u+0x777u);
    uint32_t*d_g,*d_out;
    GPU_CHECK(hipMalloc(&d_g,N*4)); GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_g,h_g,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_g,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++) if(h_out[i]!=h_g[i]){
        fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],h_g[i]);pass=0;break;}
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
