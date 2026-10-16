/* HIP test for BUFFER_ATOMIC_DEC_X2 (ENC_MUBUF, 64-bit atomic dec, return-old,
 * gfx950). Per-thread 8-byte slot via raw V#; AtomicOpDecWraparound: keep
 * 0<init<=cmp so mem=init-1, old=init. base only. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(uint64_t*mem,uint64_t cmp,uint64_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)mem;
    u32x4 srd; srd.x=(uint32_t)(base&0xffffffffu); srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu; srd.w=0x00020000u;
    uint32_t voffset=(uint32_t)(i*8); uint64_t v=cmp;
    __asm__ volatile("buffer_atomic_dec_x2 %0, %1, %2, 0 offen sc0\n\ts_waitcnt vmcnt(0)\n\t":"+v"(v):"v"(voffset),"s"(srd):"memory");
    old_out[i]=v;
}
int main(void){
    uint64_t*h_i=(uint64_t*)malloc(N*8),*h_m=(uint64_t*)malloc(N*8),*h_o=(uint64_t*)malloc(N*8);
    const uint64_t CMP=0xFFFFFFFFFFULL;
    for(int i=0;i<N;i++) h_i[i]=(uint64_t)(i+1)*1000003ULL;
    uint64_t*d_mem,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*8));GPU_CHECK(hipMalloc(&d_old,N*8));
    GPU_CHECK(hipMemcpy(d_mem,h_i,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,CMP,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_m,d_mem,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_o,d_old,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint64_t eo=h_i[i],em=h_i[i]-1;
        if(h_o[i]!=eo||h_m[i]!=em){fprintf(stderr,"FAIL[%d] old=%llu(e %llu) mem=%llu(e %llu)\n",i,(unsigned long long)h_o[i],(unsigned long long)eo,(unsigned long long)h_m[i],(unsigned long long)em);pass=0;break;}}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
