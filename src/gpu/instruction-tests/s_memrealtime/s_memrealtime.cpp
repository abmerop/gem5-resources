/*
 * HIP test for S_MEMREALTIME (SMEM): reads a 64-bit real-time counter into an
 * SGPR pair (gem5 returns curTick()/10000, a monotonic nonzero value).  Since
 * the exact value is timing-dependent, the test verifies determinism-independent
 * invariants: two successive reads are both nonzero and the second is >= the
 * first (monotonic non-decreasing).  Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
__global__ void kernel(uint64_t*out){
    if(threadIdx.x!=0) return;
    uint64_t t1,t2;
    __asm__ volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)\n\t":"=s"(t1)::"memory");
    __asm__ volatile("s_memrealtime %0\n\ts_waitcnt lgkmcnt(0)\n\t":"=s"(t2)::"memory");
    out[0]=t1; out[1]=t2;
}
int main(void){
    uint64_t*dO,ho[2]; GPU_CHECK(hipMalloc(&dO,16));
    kernel<<<1,64>>>(dO);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,16,hipMemcpyDeviceToHost));
    int pass=(ho[0]!=0 && ho[1]>=ho[0]);
    if(!pass)fprintf(stderr,"FAIL t1=%llu t2=%llu\n",(unsigned long long)ho[0],(unsigned long long)ho[1]);
    else printf("t1=%llu t2=%llu\n",(unsigned long long)ho[0],(unsigned long long)ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
