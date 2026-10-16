/* HIP test for GLOBAL_ATOMIC_PK_ADD_BF16 (32-bit VMEM packed-bf16 atomic add).
 * gem5 (decodes to Inst_FLAT__FLAT_ATOMIC_PK_ADD_BF16 -> AtomicOpPkAddBF16):
 * MEM.bf16lo += DATA.bf16lo; MEM.bf16hi += DATA.bf16hi (no value returned).
 * Per-lane distinct 4-byte global slot; small exact-int bf16 values -> bit-exact.
 * Variants exercised: base */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(uint32_t*mem,const uint32_t*data,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(&mem[i]);
    uint32_t v=data[i];
    __asm__ volatile("global_atomic_pk_add_bf16 %0, %1, off\n\ts_waitcnt vmcnt(0)\n\t"
                     ::"v"(addr),"v"(v):"memory");
}
static uint16_t bfb(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)(u>>16); }
static float bfv(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static uint32_t pack(uint16_t lo,uint16_t hi){ return (uint32_t)lo|((uint32_t)hi<<16); }
int main(void){
    uint32_t*h_i=(uint32_t*)malloc(N*4),*h_d=(uint32_t*)malloc(N*4),*h_m=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ h_i[i]=pack(bfb((float)(i%7)),bfb((float)((i%5)+1)));
        h_d[i]=pack(bfb((float)((i%3)+2)),bfb((float)(i%9))); }
    uint32_t*d_mem,*d_data;
    GPU_CHECK(hipMalloc(&d_mem,N*4));GPU_CHECK(hipMalloc(&d_data,N*4));
    GPU_CHECK(hipMemcpy(d_mem,h_i,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_d,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_m,d_mem,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint16_t ilo=h_i[i]&0xffff,ihi=h_i[i]>>16,dlo=h_d[i]&0xffff,dhi=h_d[i]>>16;
        uint32_t e=pack(bfb(bfv(ilo)+bfv(dlo)),bfb(bfv(ihi)+bfv(dhi)));
        if(h_m[i]!=e){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,h_m[i],e);pass=0;break;} }
    if(pass)printf("mem[1]=%08x\n",h_m[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
