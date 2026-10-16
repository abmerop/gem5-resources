/*
 * HIP test for BUFFER_ATOMIC_CMPSWAP_X2 (MUBUF 64-bit compare-and-swap).
 * Pattern C, base only.
 *
 * data VGPRs = { swap:64, cmp:64 } (v[0:3]); MEM64 = (MEM64==cmp) ? swap : MEM64;
 * returns old MEM64 in v[0:1].  Per-lane distinct 8-byte slot via a raw V#;
 * each slot is seeded == cmp so the swap fires.  Verifies old==init and
 * mem==swap.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint64_t u64x2 __attribute__((ext_vector_type(2)));
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(uint64_t*mem,const uint64_t*srcv,const uint64_t*cmpv,uint64_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)mem; u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu); srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu; srd.w=0x00020000u;
    u64x2 data; data.x=srcv[i]; data.y=cmpv[i];   /* {swap, cmp} */
    uint32_t voff=(uint32_t)(i*8);
    __asm__ volatile("buffer_atomic_cmpswap_x2 %0, %1, %2, 0 offen sc0\n\ts_waitcnt vmcnt(0)\n\t"
                     :"+v"(data):"v"(voff),"s"(srd):"memory");
    old_out[i]=data.x;
}
int main(void){
    uint64_t*h_i=(uint64_t*)malloc(N*8),*h_s=(uint64_t*)malloc(N*8),*h_c=(uint64_t*)malloc(N*8);
    uint64_t*h_old=(uint64_t*)malloc(N*8),*h_m=(uint64_t*)malloc(N*8);
    for(int i=0;i<N;i++){ h_i[i]=(uint64_t)(i)*0x100000001ull+7ull; h_c[i]=h_i[i];
                          h_s[i]=(uint64_t)(i)*0x00A0000B0ull+0x50000ull; }
    uint64_t*d_mem,*d_s,*d_c,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*8));GPU_CHECK(hipMalloc(&d_s,N*8));GPU_CHECK(hipMalloc(&d_c,N*8));GPU_CHECK(hipMalloc(&d_old,N*8));
    GPU_CHECK(hipMemcpy(d_mem,h_i,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_s,h_s,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_s,d_c,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_old,d_old,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_m,d_mem,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ if(h_old[i]!=h_i[i]||h_m[i]!=h_s[i]){
        fprintf(stderr,"FAIL[%d] old=%llu(e%llu) mem=%llu(e%llu)\n",i,
            (unsigned long long)h_old[i],(unsigned long long)h_i[i],
            (unsigned long long)h_m[i],(unsigned long long)h_s[i]);pass=0;break;} }
    if(pass)printf("old[0]=%llu mem[0]=%llu\n",(unsigned long long)h_old[0],(unsigned long long)h_m[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
