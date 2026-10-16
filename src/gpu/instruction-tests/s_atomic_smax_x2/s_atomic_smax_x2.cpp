/*
 * HIP test for S_ATOMIC_SMAX_X2 (SMEM scalar 64-bit atomic, gfx950/CDNA4).
 * One wavefront per counter; wavefront-uniform data value (SGPR).
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_atomic_smax_x2 s_atomic_smax_x2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NUM_WF=64;
__global__ void kernel(int64_t*counters,int64_t*old_out,int64_t dval){
    int wf=blockIdx.x; int64_t*slot=&counters[wf];
    int64_t old=dval;
    __asm__ volatile("s_atomic_smax_x2 %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)\n\t":"+s"(old):"s"(slot):"memory");
    if(threadIdx.x==0) old_out[wf]=old;
}
int main(void){
    int64_t*h_init=(int64_t*)malloc(NUM_WF*sizeof(int64_t)),*h_mem=(int64_t*)malloc(NUM_WF*sizeof(int64_t)),*h_old=(int64_t*)malloc(NUM_WF*sizeof(int64_t));
    const int64_t DVAL=(int64_t)0x0000000100000007LL;
    for(int w=0;w<NUM_WF;w++) h_init[w]=((int64_t)((w%2)? -(int64_t)(w*7919+11) : (int64_t)(w*104729+3)));
    int64_t*d_c,*d_old;
    GPU_CHECK(hipMalloc(&d_c,NUM_WF*sizeof(int64_t)));GPU_CHECK(hipMalloc(&d_old,NUM_WF*sizeof(int64_t)));
    GPU_CHECK(hipMemcpy(d_c,h_init,NUM_WF*sizeof(int64_t),hipMemcpyHostToDevice));
    kernel<<<NUM_WF,64>>>(d_c,d_old,DVAL);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_c,NUM_WF*sizeof(int64_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,NUM_WF*sizeof(int64_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int w=0;w<NUM_WF;w++){
        int64_t eo=h_init[w]; int64_t em=(h_init[w] > (DVAL) ? h_init[w] : (DVAL));
        if(h_old[w]!=eo||h_mem[w]!=em){
            fprintf(stderr,"FAIL[%d] old=%lld(exp %lld) mem=%lld(exp %lld)\n",w,(long long)h_old[w],(long long)eo,(long long)h_mem[w],(long long)em);
            pass=0;break;}
    }
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
