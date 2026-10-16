/*
 * HIP test for FLAT_ATOMIC_AND (ENC_FLAT, 32-bit bitwise-AND atomic,
 * return-old).
 *
 *   tmp       = MEM[addr]
 *   MEM[addr] = tmp & DATA
 *   VDST      = tmp     (old value, sc0 return)
 *
 * Each thread targets a disjoint slot (no contention).  Pattern C, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_atomic_and flat_atomic_and.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kernel(uint32_t *mem, const uint32_t *data, uint32_t *old_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t addr = (uint64_t)(mem + i);
    uint32_t v = data[i], old;
    __asm__ volatile(
        "flat_atomic_and %0, %1, %2 sc0\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        : "=v"(old) : "v"(addr), "v"(v) : "memory");
    old_out[i] = old;
}

int main(void)
{
    uint32_t *h_init=(uint32_t*)malloc(N*4),*h_data=(uint32_t*)malloc(N*4);
    uint32_t *h_mem=(uint32_t*)malloc(N*4),*h_old=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ h_init[i]=(uint32_t)(i*2654435761u); h_data[i]=(uint32_t)(0xf0f0f0f0u ^ (i*40503u)); }
    uint32_t *d_mem,*d_data,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*4));GPU_CHECK(hipMalloc(&d_data,N*4));GPU_CHECK(hipMalloc(&d_old,N*4));
    GPU_CHECK(hipMemcpy(d_mem,h_init,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_data,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_mem,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t eo=h_init[i], em=h_init[i]&h_data[i];
        if(h_old[i]!=eo||h_mem[i]!=em){fprintf(stderr,"FAIL[%d] old=0x%x(exp 0x%x) mem=0x%x(exp 0x%x)\n",i,h_old[i],eo,h_mem[i],em);pass=0;break;}
    }
    if(pass)printf("old[0]=0x%x mem[0]=0x%x\n",h_old[0],h_mem[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
