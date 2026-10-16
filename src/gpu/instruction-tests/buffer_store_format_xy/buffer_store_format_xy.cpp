/*
 * HIP test for BUFFER_STORE_FORMAT_XY (ENC_MUBUF, typed 2-component store).
 * Pattern C, base only.
 *
 * BUFFER_STORE_FORMAT_XY writes two components {x,y} from v[0:1] through a
 * buffer resource descriptor.  Using a 32_32 / UINT format the two dwords pass
 * through unconverted, so the host can read the buffer back and verify both
 * components per element.  The descriptor is built over the output array
 * (offen, per-thread voffset).  Placed via inline asm since HIP does not emit
 * format stores for plain global writes.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   buffer_store_format_xy v[0:1], v2, s[0:3], 0 offen
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o buffer_store_format_xy \
 *         buffer_store_format_xy.cpp
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

__global__ void kernel(uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)out;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    /* dst_sel identity; dfmt=32_32(11), nfmt=UINT(4). */
    srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(11u<<12)|(4u<<15);
    u32x2 data;
    data.x=(uint32_t)(i*2654435761u);
    data.y=(uint32_t)(i*40503u+7u);
    uint32_t voffset=(uint32_t)(i*8);
    __asm__ volatile(
        "buffer_store_format_xy %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        ::"v"(data),"v"(voffset),"s"(srd):"memory");
}
int main(void){
    uint32_t*h_out=(uint32_t*)malloc(N*8);
    for(int i=0;i<2*N;i++) h_out[i]=0u;
    uint32_t*d_out;
    GPU_CHECK(hipMalloc(&d_out,N*8));
    GPU_CHECK(hipMemcpy(d_out,h_out,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t ex=(uint32_t)(i*2654435761u), ey=(uint32_t)(i*40503u+7u);
        if(h_out[2*i]!=ex||h_out[2*i+1]!=ey){
            fprintf(stderr,"FAIL[%d] got 0x%x,0x%x exp 0x%x,0x%x\n",
                i,h_out[2*i],h_out[2*i+1],ex,ey);pass=0;break;}}
    if(pass)printf("out0=0x%x,0x%x\n",h_out[0],h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
