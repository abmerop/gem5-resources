/*
 * HIP test for BUFFER_ATOMIC_CMPSWAP (MUBUF 32-bit compare-and-swap).
 * data VGPR pair = {src, cmp}: MEM = (MEM==cmp) ? src : MEM; returns old MEM.
 * (The 32-bit CAS path is correct in gem5, unlike the buggy 64-bit _x2 path.)
 * Per-lane distinct 4-byte slot via raw V#; mem seeded == cmp so the swap fires;
 * verifies old==init and mem==src.  Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(uint32_t*mem,const uint32_t*srcv,const uint32_t*cmpv,uint32_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)mem; u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu); srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu; srd.w=0x00020000u;
    u32x2 data; data.x=srcv[i]; data.y=cmpv[i];
    uint32_t voff=(uint32_t)(i*4);
    __asm__ volatile("buffer_atomic_cmpswap %0, %1, %2, 0 offen sc0\n\ts_waitcnt vmcnt(0)\n\t"
                     :"+v"(data):"v"(voff),"s"(srd):"memory");
    old_out[i]=data.x;
}
int main(void){
    uint32_t*h_i=(uint32_t*)malloc(N*4),*h_s=(uint32_t*)malloc(N*4),*h_c=(uint32_t*)malloc(N*4);
    uint32_t*h_old=(uint32_t*)malloc(N*4),*h_m=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ h_i[i]=(uint32_t)(i*1000+7); h_c[i]=h_i[i]; h_s[i]=(uint32_t)(i*7+0x50000); }
    uint32_t*d_mem,*d_s,*d_c,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*4));GPU_CHECK(hipMalloc(&d_s,N*4));GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_old,N*4));
    GPU_CHECK(hipMemcpy(d_mem,h_i,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_s,h_s,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_s,d_c,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_old,d_old,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_m,d_mem,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ if(h_old[i]!=h_i[i]||h_m[i]!=h_s[i]){
        fprintf(stderr,"FAIL[%d] old=%u(e%u) mem=%u(e%u)\n",i,h_old[i],h_i[i],h_m[i],h_s[i]);pass=0;break;} }
    if(pass)printf("old[0]=%u mem[0]=%u\n",h_old[0],h_m[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
