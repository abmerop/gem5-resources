/*
 * HIP test for S_BUFFER_ATOMIC_UMAX_X2 (ENC_SMEM 64-bit scalar buffer atomic, gfx950).
 * Single wavefront, one 64-bit slot via a raw buffer descriptor (V# in SGPRs).
 * Host reference matches gem5 atomic functor. Build: hipcc -O2 --offload-arch=gfx950.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int BLOCK_SIZE=64;
__global__ void kernel(uint64_t*mem,uint64_t addval,uint64_t*old_out){
    uint64_t base=(uint64_t)mem; u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu); srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu; srd.w=0x00020000u;
    uint64_t d=addval;
    __asm__ volatile("s_buffer_atomic_umax_x2 %0, %1, 0 glc\n\ts_waitcnt lgkmcnt(0)\n\t":"+s"(d):"s"(srd):"memory");
    if(threadIdx.x==0) old_out[0]=d;
}
int main(void){
    const uint64_t init=0x0000000100000064ULL; const uint64_t addval=0x0000000200000007ULL;
    uint64_t h_mem=0,h_old=0; uint64_t*d_mem,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,sizeof(uint64_t)));GPU_CHECK(hipMalloc(&d_old,sizeof(uint64_t)));
    GPU_CHECK(hipMemcpy(d_mem,&init,sizeof(uint64_t),hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel,dim3(1),dim3(BLOCK_SIZE),0,0,d_mem,addval,d_old);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(&h_mem,d_mem,sizeof(uint64_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(&h_old,d_old,sizeof(uint64_t),hipMemcpyDeviceToHost));
    uint64_t exp_old=init; uint64_t exp_mem=(init > addval ? init : addval);
    int pass=(h_old==exp_old)&&(h_mem==exp_mem);
    if(!pass) fprintf(stderr,"FAIL old=%llu(exp %llu) mem=%llu(exp %llu)\n",(unsigned long long)h_old,(unsigned long long)exp_old,(unsigned long long)h_mem,(unsigned long long)exp_mem);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_mem));GPU_CHECK(hipFree(d_old));
    return pass?0:1;
}
