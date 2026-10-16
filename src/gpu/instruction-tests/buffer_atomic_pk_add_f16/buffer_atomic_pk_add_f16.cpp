/* HIP test for BUFFER_ATOMIC_PK_ADD_F16 (MUBUF packed-f16 atomic add).
 * gem5 (Inst_MUBUF__BUFFER_ATOMIC_PK_ADD_F16 -> AtomicOpPkAddF16):
 * MEM.f16lo += DATA.f16lo; MEM.f16hi += DATA.f16hi (no value returned).
 * Per-lane distinct 4-byte slot via raw V#; small exact-int f16 -> bit-exact.
 * Variants exercised: base */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(uint32_t*mem,const uint32_t*data,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)mem; u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu); srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu; srd.w=0x00020000u;
    uint32_t v=data[i]; uint32_t voff=(uint32_t)(i*4);
    __asm__ volatile("buffer_atomic_pk_add_f16 %0, %1, %2, 0 offen\n\ts_waitcnt vmcnt(0)\n\t"
                     ::"v"(v),"v"(voff),"s"(srd):"memory");
}
static uint16_t hb(float v){ __half h=__float2half(v); uint16_t b; __builtin_memcpy(&b,&h,2); return b; }
static float hv(uint16_t b){ __half h; __builtin_memcpy(&h,&b,2); return __half2float(h); }
static uint32_t pack(uint16_t lo,uint16_t hi){ return (uint32_t)lo|((uint32_t)hi<<16); }
int main(void){
    uint32_t*h_i=(uint32_t*)malloc(N*4),*h_d=(uint32_t*)malloc(N*4),*h_m=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ h_i[i]=pack(hb((float)(i%7)),hb((float)((i%5)+1)));
        h_d[i]=pack(hb((float)((i%3)+2)),hb((float)(i%9))); }
    uint32_t*d_mem,*d_data;
    GPU_CHECK(hipMalloc(&d_mem,N*4));GPU_CHECK(hipMalloc(&d_data,N*4));
    GPU_CHECK(hipMemcpy(d_mem,h_i,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_d,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_m,d_mem,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint16_t ilo=h_i[i]&0xffff,ihi=h_i[i]>>16,dlo=h_d[i]&0xffff,dhi=h_d[i]>>16;
        uint32_t e=pack(hb(hv(ilo)+hv(dlo)),hb(hv(ihi)+hv(dhi)));
        if(h_m[i]!=e){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,h_m[i],e);pass=0;break;} }
    if(pass)printf("mem[1]=%08x\n",h_m[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
