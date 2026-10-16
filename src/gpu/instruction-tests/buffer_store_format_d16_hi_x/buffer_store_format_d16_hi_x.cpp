/*
 * HIP test for BUFFER_STORE_FORMAT_D16_HI_X (ENC_MUBUF, typed 1-component
 * store from the HIGH 16 bits of a VGPR).  Pattern C, base only.
 *
 * BUFFER_STORE_FORMAT_D16_HI_X stores the upper halfword of the source VGPR as
 * one 16-bit component {x} to a buffer resource.  With data format 16 / UINT
 * this is an identity write of the high halfword.  Each thread places a known
 * value in the high halfword, stores it, and the host reads the buffer back
 * and compares.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   buffer_store_format_d16_hi_x v0, v2, s[0:3], 0 offen
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o buffer_store_format_d16_hi_x \
 *         buffer_store_format_d16_hi_x.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256, BLOCK_SIZE=64;

__global__ void kernel(const uint16_t*src,uint16_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)out;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    /* dst_sel identity; nfmt=UINT(4), dfmt=16(2). */
    srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(2u<<12)|(4u<<15);
    uint32_t voffset=(uint32_t)(i*2);
    uint32_t data=((uint32_t)src[i]<<16)|0x1234u;   /* value in high half */
    __asm__ volatile(
        "buffer_store_format_d16_hi_x %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :: "v"(data),"v"(voffset),"s"(srd):"memory");
}
int main(void){
    uint16_t*h_src=(uint16_t*)malloc(N*2);
    uint16_t*h_out=(uint16_t*)malloc(N*2);
    for(int i=0;i<N;i++){h_src[i]=(uint16_t)(i*1103+7);h_out[i]=0;}
    uint16_t*d_src,*d_out;
    GPU_CHECK(hipMalloc(&d_src,N*2));GPU_CHECK(hipMalloc(&d_out,N*2));
    GPU_CHECK(hipMemcpy(d_src,h_src,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_out,0,N*2));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*2,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){if(h_out[i]!=h_src[i]){
        fprintf(stderr,"FAIL[%d] got 0x%x exp 0x%x\n",i,h_out[i],h_src[i]);
        pass=0;break;}}
    if(pass)printf("out[0]=0x%x out[%d]=0x%x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
