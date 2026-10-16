/*
 * HIP test for BUFFER_LOAD_FORMAT_D16_XYZW (ENC_MUBUF, typed 4-component d16
 * load).  Pattern C, base only.
 *
 * BUFFER_LOAD_FORMAT_D16_XYZW loads four 16-bit components {x,y,z,w} from a
 * buffer resource, packing them into two consecutive VGPRs (x,y in dword0;
 * z,w in dword1).  With a 16_16_16_16 / UINT format the halfwords pass through
 * unconverted, so the host seeds a u16[4] buffer and verifies each component.
 * Placed via inline asm (offen, per-thread voffset) since HIP does not emit
 * format loads for plain global reads.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   buffer_load_format_d16_xyzw v[0:1], v2, s[0:3], 0 offen
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o buffer_load_format_d16_xyzw \
 *         buffer_load_format_d16_xyzw.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
static const int N=256, BLOCK_SIZE=64;

__global__ void kernel(const uint16_t*src,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)src;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    /* dst_sel identity; dfmt=16_16_16_16(12), nfmt=UINT(4). */
    srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(12u<<12)|(4u<<15);
    uint32_t voffset=(uint32_t)(i*8);   /* 4 * u16 per element */
    u32x2 data={0u,0u};
    __asm__ volatile(
        "buffer_load_format_d16_xyzw %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :"+v"(data):"v"(voffset),"s"(srd):"memory");
    out[2*i]=data.x; out[2*i+1]=data.y;
}
int main(void){
    uint16_t*h_src=(uint16_t*)malloc(N*4*2);
    uint32_t*h_out=(uint32_t*)malloc(N*8);
    for(int i=0;i<4*N;i++) h_src[i]=(uint16_t)(i*1103+7);
    uint16_t*d_src; uint32_t*d_out;
    GPU_CHECK(hipMalloc(&d_src,N*4*2));GPU_CHECK(hipMalloc(&d_out,N*8));
    GPU_CHECK(hipMemcpy(d_src,h_src,N*4*2,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint16_t x=h_src[4*i],y=h_src[4*i+1],z=h_src[4*i+2],w=h_src[4*i+3];
        uint32_t exlo=((uint32_t)y<<16)|x, exhi=((uint32_t)w<<16)|z;
        if(h_out[2*i]!=exlo||h_out[2*i+1]!=exhi){
            fprintf(stderr,"FAIL[%d] got 0x%x,0x%x exp 0x%x,0x%x\n",
                i,h_out[2*i],h_out[2*i+1],exlo,exhi);pass=0;break;}}
    if(pass)printf("out0=0x%x,0x%x\n",h_out[0],h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
