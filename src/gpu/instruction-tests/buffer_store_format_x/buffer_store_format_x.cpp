/*
 * HIP test for BUFFER_STORE_FORMAT_X (ENC_MUBUF, typed 1-component store).
 *
 * Stores one component from a VGPR to a buffer resource at a per-thread byte
 * offset (offen), using a FORMAT_32_UINT resource descriptor so the stored
 * value is bit-preserved.  The result is read back via hipMemcpy.  Pattern C,
 * base only (MUBUF has no DPP/SDWA).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o buffer_store_format_x buffer_store_format_x.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256, BLOCK_SIZE=64;

__global__ void kernel(uint32_t*dst,const uint32_t*vals,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint64_t base=(uint64_t)dst;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(4u<<12)|(4u<<15); /* dfmt=32,nfmt=uint */
    uint32_t voffset=(uint32_t)(i*4);
    uint32_t data=vals[i];
    __asm__ volatile(
        "buffer_store_format_x %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        ::"v"(data),"v"(voffset),"s"(srd):"memory");
}
int main(void){
    uint32_t*h_vals=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_vals[i]=(uint32_t)(i*2654435761u+12345u);
    uint32_t*d_dst,*d_vals;
    GPU_CHECK(hipMalloc(&d_dst,N*4));GPU_CHECK(hipMalloc(&d_vals,N*4));
    GPU_CHECK(hipMemset(d_dst,0,N*4));
    GPU_CHECK(hipMemcpy(d_vals,h_vals,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_dst,d_vals,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_dst,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++) if(h_out[i]!=h_vals[i]){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],h_vals[i]);pass=0;break;}
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
