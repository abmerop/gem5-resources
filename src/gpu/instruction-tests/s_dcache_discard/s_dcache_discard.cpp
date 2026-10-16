/*
 * HIP test for S_DCACHE_DISCARD (ENC_SMEM, discard a scalar data cache line).
 * Pattern B, base only.
 *
 * S_DCACHE_DISCARD marks the scalar data cache line for the address in the
 * SGPR base pair (plus offset) as invalid without writing it back.  It has no
 * architectural result.  This test issues a scalar store to a global slot,
 * then S_DCACHE_DISCARD against a scratch address, then verifies the store
 * from the host.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   s_dcache_discard s[2:3], 0
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o s_dcache_discard s_dcache_discard.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NWAVES=64;

__global__ void kernel(unsigned long long*out,unsigned long long*scratch){
    int b=blockIdx.x;
    unsigned long long*p=out+b;
    unsigned long long*s=scratch+b;
    unsigned long long val=0x99AABBCC00000000ULL+(unsigned long long)b;
    __asm__ volatile(
        "s_store_dwordx2 %0, %1, 0x0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        "s_dcache_discard %2, 0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :: "s"(val),"s"(p),"s"(s):"memory");
}
int main(void){
    unsigned long long*h=(unsigned long long*)malloc(NWAVES*8);
    for(int i=0;i<NWAVES;i++)h[i]=0;
    unsigned long long*d,*sc;
    GPU_CHECK(hipMalloc(&d,NWAVES*8));GPU_CHECK(hipMemset(d,0,NWAVES*8));
    GPU_CHECK(hipMalloc(&sc,NWAVES*8));GPU_CHECK(hipMemset(sc,0,NWAVES*8));
    hipLaunchKernelGGL(kernel,dim3(NWAVES),dim3(1),0,0,d,sc);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h,d,NWAVES*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<NWAVES;i++){unsigned long long e=0x99AABBCC00000000ULL+i;
        if(h[i]!=e){fprintf(stderr,"FAIL[%d] got %016llx exp %016llx\n",i,h[i],e);
            pass=0;break;}}
    if(pass)printf("out[0]=%016llx out[%d]=%016llx\n",h[0],NWAVES-1,h[NWAVES-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
