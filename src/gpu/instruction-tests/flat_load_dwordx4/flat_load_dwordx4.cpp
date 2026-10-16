/*
 * HIP test for FLAT_LOAD_DWORDX4 (ENC_FLAT, 4-dword flat load).
 *
 *   VDST[0:3] = MEM[addr .. addr+15]
 *
 * Each thread loads 16 bytes from a disjoint slot.  Pattern C, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_load_dwordx4 flat_load_dwordx4.cpp
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
static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kernel(const uint32_t *src, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t addr = (uint64_t)(src + i*4);
    u32x4 dst;
    __asm__ volatile(
        "flat_load_dwordx4 %0, %1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(dst) : "v"(addr) : "memory");
    out[i*4+0]=dst.x; out[i*4+1]=dst.y; out[i*4+2]=dst.z; out[i*4+3]=dst.w;
}

int main(void)
{
    const int total=N*4;
    uint32_t *h_src=(uint32_t*)malloc(total*4),*h_out=(uint32_t*)malloc(total*4);
    for(int i=0;i<total;i++) h_src[i]=(uint32_t)(i*2654435761u+99u);
    uint32_t *d_src,*d_out;
    GPU_CHECK(hipMalloc(&d_src,total*4));GPU_CHECK(hipMalloc(&d_out,total*4));
    GPU_CHECK(hipMemcpy(d_src,h_src,total*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,total*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<total;i++){ if(h_out[i]!=h_src[i]){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],h_src[i]);pass=0;break;} }
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],total-1,h_out[total-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
