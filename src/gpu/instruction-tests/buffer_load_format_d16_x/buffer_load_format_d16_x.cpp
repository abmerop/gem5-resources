/*
 * HIP test for BUFFER_LOAD_FORMAT_D16_X (ENC_MUBUF, typed 1-component buffer
 * load into a packed 16-bit destination).  Pattern C, base only.
 *
 * Loads one 16-bit component {x} from a buffer resource into the low 16 bits
 * of a VGPR.  The buffer resource descriptor (V#) is built over a 16-bit
 * source allocation with a 16 UINT format; a per-thread byte offset (offen)
 * selects each 2-byte record.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o buffer_load_format_d16_x \
 *         buffer_load_format_d16_x.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256, BLOCK_SIZE=64;
__global__ void kernel(const uint16_t*src,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)src;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    /* dst_sel identity; nfmt=UINT(4), dfmt=16(2). */
    srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(2u<<12)|(4u<<15);
    uint32_t voffset=(uint32_t)(i*2);       /* per-thread 2B slot */
    uint32_t data;
    __asm__ volatile(
        "buffer_load_format_d16_x %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :"=v"(data):"v"(voffset),"s"(srd):"memory");
    out[i]=data&0xffffu;
}
int main(void){
    uint16_t*h_src=(uint16_t*)malloc(N*2);
    uint32_t*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_src[i]=(uint16_t)(i*1103+7);
    uint16_t*d_src; uint32_t*d_out;
    GPU_CHECK(hipMalloc(&d_src,N*2));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_src,h_src,N*2,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t exp=(uint32_t)h_src[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%x exp 0x%x\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%x out[%d]=0x%x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
