/*
 * HIP test for S_BUFFER_ATOMIC_DEC_X2 (SMEM 64-bit scalar-buffer atomic dec).
 * gem5 uses the wraparound dec functor: new = (old==0 || old>cmp) ? cmp
 * : old-1, and returns old.  64-bit scalar-buffer atomics work (unlike the
 * broken 32-bit path).  Single wave, one 64-bit slot; cmp is the data operand
 * (glc return).  init=7, cmp=100 -> old=7, mem=6.  Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

__global__ void kernel(uint64_t*mem,uint64_t cmp,uint64_t*old_out){
    uint64_t base=(uint64_t)mem; u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu); srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu; srd.w=0x00020000u;
    uint64_t d=cmp;
    __asm__ volatile("s_buffer_atomic_dec_x2 %0, %1, 0 glc\n\ts_waitcnt lgkmcnt(0)\n\t"
                     :"+s"(d):"s"(srd):"memory");
    if(threadIdx.x==0) old_out[0]=d;
}
int main(void){
    uint64_t init=7, cmp=100;
    uint64_t *d_mem,*d_old, h_mem, h_old;
    GPU_CHECK(hipMalloc(&d_mem,8));GPU_CHECK(hipMalloc(&d_old,8));
    GPU_CHECK(hipMemcpy(d_mem,&init,8,hipMemcpyHostToDevice));
    kernel<<<1,64>>>(d_mem,cmp,d_old);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(&h_old,d_old,8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(&h_mem,d_mem,8,hipMemcpyDeviceToHost));
    uint64_t exp_old=7, exp_mem=6;
    int pass=(h_old==exp_old && h_mem==exp_mem);
    if(!pass) fprintf(stderr,"FAIL old=%llu(e%llu) mem=%llu(e%llu)\n",
        (unsigned long long)h_old,(unsigned long long)exp_old,
        (unsigned long long)h_mem,(unsigned long long)exp_mem);
    else printf("old=%llu mem=%llu\n",(unsigned long long)h_old,(unsigned long long)h_mem);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
