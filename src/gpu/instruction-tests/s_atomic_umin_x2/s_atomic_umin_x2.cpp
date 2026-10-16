/*
 * HIP test for S_ATOMIC_UMIN_X2 (ENC_SMEM, 64-bit scalar atomic unsigned min,
 * return-old).  Pattern B.
 *
 *   tmp = MEM[addr]                       (64-bit)
 *   MEM[addr] = umin(tmp, SDATA)          (64-bit unsigned min)
 *   if (GLC) SDATA = tmp                  (old value)
 *
 * SDATA is in/out ("+s" on a 64-bit value = SGPR pair); the address is a
 * 64-bit SGPR pair.  Each wavefront (one per block) targets its own slot, so
 * the result is contention-free and deterministic.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_atomic_umin_x2 s_atomic_umin_x2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NUM_WF=8,BLOCK_SIZE=64;
__global__ void kernel(uint64_t*counters,const uint64_t*data,uint64_t*old_out){
    int wf=blockIdx.x;
    uint64_t*slot=&counters[wf];
    uint64_t v=data[wf];
    __asm__ volatile(
        "s_atomic_umin_x2 %0, %1, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"+s"(v):"s"(slot):"memory");
    if(threadIdx.x==0) old_out[wf]=v;
}
int main(void){
    uint64_t*h_init=(uint64_t*)malloc(NUM_WF*8),*h_data=(uint64_t*)malloc(NUM_WF*8);
    uint64_t*h_mem=(uint64_t*)malloc(NUM_WF*8),*h_old=(uint64_t*)malloc(NUM_WF*8);
    for(int i=0;i<NUM_WF;i++){
        h_init[i]=((uint64_t)(i+1)<<32)|(uint64_t)(100-i*7);
        /* alternate so min picks data sometimes, init other times */
        h_data[i]=(i&1)?(h_init[i]+0x100000000ULL):(h_init[i]-1);
    }
    uint64_t*d_cnt,*d_data,*d_old;
    GPU_CHECK(hipMalloc(&d_cnt,NUM_WF*8));GPU_CHECK(hipMalloc(&d_data,NUM_WF*8));GPU_CHECK(hipMalloc(&d_old,NUM_WF*8));
    GPU_CHECK(hipMemcpy(d_cnt,h_init,NUM_WF*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_data,NUM_WF*8,hipMemcpyHostToDevice));
    kernel<<<NUM_WF,BLOCK_SIZE>>>(d_cnt,d_data,d_old);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_cnt,NUM_WF*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,NUM_WF*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<NUM_WF;i++){
        uint64_t exp_old=h_init[i];
        uint64_t exp_mem=h_init[i]<h_data[i]?h_init[i]:h_data[i];
        if(h_old[i]!=exp_old||h_mem[i]!=exp_mem){
            fprintf(stderr,"FAIL[%d] old=%llx(exp %llx) mem=%llx(exp %llx)\n",i,
                (unsigned long long)h_old[i],(unsigned long long)exp_old,
                (unsigned long long)h_mem[i],(unsigned long long)exp_mem);pass=0;break;}
    }
    if(pass)printf("old[1]=%llx mem[1]=%llx\n",(unsigned long long)h_old[1],(unsigned long long)h_mem[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
