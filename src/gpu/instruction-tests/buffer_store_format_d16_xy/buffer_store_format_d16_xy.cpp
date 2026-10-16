/*
 * HIP test for BUFFER_STORE_FORMAT_D16_XY (ENC_MUBUF, typed 2-component
 * packed-16 buffer store).  Pattern C, base only.
 *
 * BUFFER_STORE_FORMAT_D16_XY stores two 16-bit components {x,y} packed into a
 * single source VGPR to a buffer resource.  With the buffer's data format set
 * to 16_16 and numeric format UINT the store is an identity write of the two
 * packed halfwords.  This test packs two u16 values per thread, stores them
 * via the instruction, and the host reads the buffer back and compares.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   buffer_store_format_d16_xy v0, v2, s[0:3], 0 offen
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o buffer_store_format_d16_xy \
 *         buffer_store_format_d16_xy.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256, BLOCK_SIZE=64;

/* Each thread packs two u16 into one VGPR and stores via the d16_xy format
 * store to its own 4-byte record (offen). */
__global__ void kernel(const uint32_t*src,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)out;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    /* dst_sel identity; nfmt=UINT(4), dfmt=16_16(5). */
    srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(5u<<12)|(4u<<15);
    uint32_t voffset=(uint32_t)(i*4);
    uint32_t data=src[i];               /* two packed u16 */
    __asm__ volatile(
        "buffer_store_format_d16_xy %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :: "v"(data),"v"(voffset),"s"(srd):"memory");
}
int main(void){
    uint32_t*h_src=(uint32_t*)malloc(N*4);
    uint32_t*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_src[i]=((uint32_t)(uint16_t)(i*7+1))|
        ((uint32_t)(uint16_t)(i*13+5)<<16); h_out[i]=0;}
    uint32_t*d_src,*d_out;
    GPU_CHECK(hipMalloc(&d_src,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_src,h_src,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_out,0,N*4));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){if(h_out[i]!=h_src[i]){
        fprintf(stderr,"FAIL[%d] got 0x%x exp 0x%x\n",i,h_out[i],h_src[i]);
        pass=0;break;}}
    if(pass)printf("out[0]=0x%x out[%d]=0x%x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
