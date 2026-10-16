/* HIP test for S_ATOMIC_DEC_X2 (SMEM 64-bit scalar atomic dec, gfx950).
 * gem5 AtomicOpDecWraparound: mem=(init==0||init>cmp)?cmp:init-1. Keep
 * 0<init<=cmp so mem=init-1, old=init. base only. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NUM_WF=64;
__global__ void kernel(uint64_t*counters,uint64_t*old_out,uint64_t cmp){
    int wf=blockIdx.x; uint64_t*slot=&counters[wf];
    uint64_t old=cmp;
    __asm__ volatile("s_atomic_dec_x2 %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)\n\t":"+s"(old):"s"(slot):"memory");
    if(threadIdx.x==0) old_out[wf]=old;
}
int main(void){
    uint64_t h_init[NUM_WF],h_mem[NUM_WF],h_old[NUM_WF];
    const uint64_t CMP=0xFFFFFFFFFFULL;
    for(int w=0;w<NUM_WF;w++) h_init[w]=(uint64_t)(w+1)*1000003ULL;
    uint64_t*d_c,*d_old;
    GPU_CHECK(hipMalloc(&d_c,NUM_WF*sizeof(uint64_t)));GPU_CHECK(hipMalloc(&d_old,NUM_WF*sizeof(uint64_t)));
    GPU_CHECK(hipMemcpy(d_c,h_init,NUM_WF*sizeof(uint64_t),hipMemcpyHostToDevice));
    kernel<<<NUM_WF,64>>>(d_c,d_old,CMP);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_c,NUM_WF*sizeof(uint64_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,NUM_WF*sizeof(uint64_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int w=0;w<NUM_WF;w++){
        uint64_t eo=h_init[w],em=h_init[w]-1;
        if(h_old[w]!=eo||h_mem[w]!=em){fprintf(stderr,"FAIL[%d] old=%llu(e %llu) mem=%llu(e %llu)\n",w,(unsigned long long)h_old[w],(unsigned long long)eo,(unsigned long long)h_mem[w],(unsigned long long)em);pass=0;break;}
    }
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
