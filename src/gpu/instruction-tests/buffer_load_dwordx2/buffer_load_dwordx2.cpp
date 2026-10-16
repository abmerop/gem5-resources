/*
 * HIP test for BUFFER_LOAD_DWORDX2 (ENC_MUBUF, untyped 2-dword buffer load).
 *
 * Loads 8 bytes (2 consecutive dwords) from a buffer resource into a VGPR pair.
 * A raw buffer resource descriptor (V#) is built over the source allocation;
 * a per-thread byte offset (offen) selects each 8-byte record.  Format fields
 * are irrelevant for untyped loads.  Pattern C, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o buffer_load_dwordx2 buffer_load_dwordx2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kernel_buffer_load_dwordx2(const uint32_t *src, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t base = (uint64_t)src;
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);
    srd.y = (uint32_t)((base >> 32) & 0xffffu);
    srd.z = 0xffffffffu;
    srd.w = (4u) | (5u << 3) | (6u << 6) | (7u << 9) | (4u << 12) | (4u << 15);
    uint32_t voffset = (uint32_t)(i * 8);
    u32x2 data;
    __asm__ volatile(
        "buffer_load_dwordx2 %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(data) : "v"(voffset), "s"(srd) : "memory");
    out[i*2+0] = data.x;
    out[i*2+1] = data.y;
}

int main(void)
{
    const int total=N*2;
    uint32_t *h_src=(uint32_t*)malloc(total*4),*h_out=(uint32_t*)malloc(total*4);
    for(int i=0;i<total;i++) h_src[i]=(uint32_t)(i*2654435761u+12345u);
    uint32_t *d_src,*d_out;
    GPU_CHECK(hipMalloc(&d_src,total*4));GPU_CHECK(hipMalloc(&d_out,total*4));
    GPU_CHECK(hipMemcpy(d_src,h_src,total*4,hipMemcpyHostToDevice));
    kernel_buffer_load_dwordx2<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,total*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<total;i++){
        if(h_out[i]!=h_src[i]){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],h_src[i]);pass=0;break;}
    }
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],total-1,h_out[total-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
