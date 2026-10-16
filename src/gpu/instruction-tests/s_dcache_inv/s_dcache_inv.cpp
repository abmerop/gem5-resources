/*
 * HIP test for S_DCACHE_INV (ENC_SMEM, scalar data-cache invalidate) on
 * CDNA4 (gfx950).
 *
 * S_DCACHE_INV invalidates the scalar data cache; it produces no register
 * result, so a correct implementation is observationally a no-op that must
 * let the wave continue.  The test emits it mid-kernel and verifies every
 * lane reaches its store afterwards.  If gem5 cannot execute S_DCACHE_INV
 * the kernel aborts and no PASSED prints.
 *
 * Base encoding only (SMEM has no DPP/SDWA).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_dcache_inv s_dcache_inv.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kernel(uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    __asm__ volatile("s_dcache_inv\n\t":::"memory");
    out[i]=(uint32_t)(i+1);
}
int main(void){
    uint32_t*h_out=(uint32_t*)malloc(N*4);
    uint32_t*d_out; GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemset(d_out,0,N*4));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++) if(h_out[i]!=(uint32_t)(i+1)){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,h_out[i],i+1);pass=0;break;}
    if(pass)printf("out[0]=%u out[%d]=%u\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
