/*
 * HIP test for S_DCACHE_WB_VOL (ENC_SMEM, write back volatile scalar cache
 * lines).  Pattern B, base only.
 *
 * S_DCACHE_WB_VOL writes back dirty volatile lines in the scalar data cache.
 * It takes no operands and produces no architectural result, so this test
 * exercises it for side effect: each wavefront performs a scalar store to its
 * own slot of a global array, issues S_DCACHE_WB_VOL to force the write back,
 * then s_waitcnt drains the pipe.  The host reads the array and verifies the
 * stored values -- if the writeback path works the data is visible.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   s_dcache_wb_vol
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o s_dcache_wb_vol s_dcache_wb_vol.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NWAVES=64;

__global__ void kernel(unsigned long long*out){
    int b=blockIdx.x;
    unsigned long long*p=out+b;
    unsigned long long val=0x5566778800000000ULL+(unsigned long long)b;
    __asm__ volatile(
        "s_store_dwordx2 %0, %1, 0x0\n\t"
        "s_dcache_wb_vol\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :: "s"(val),"s"(p):"memory");
}
int main(void){
    unsigned long long*h=(unsigned long long*)malloc(NWAVES*8);
    for(int i=0;i<NWAVES;i++)h[i]=0;
    unsigned long long*d;
    GPU_CHECK(hipMalloc(&d,NWAVES*8));GPU_CHECK(hipMemset(d,0,NWAVES*8));
    hipLaunchKernelGGL(kernel,dim3(NWAVES),dim3(1),0,0,d);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h,d,NWAVES*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<NWAVES;i++){unsigned long long e=0x5566778800000000ULL+i;
        if(h[i]!=e){fprintf(stderr,"FAIL[%d] got %016llx exp %016llx\n",i,h[i],e);
            pass=0;break;}}
    if(pass)printf("out[0]=%016llx out[%d]=%016llx\n",h[0],NWAVES-1,h[NWAVES-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
