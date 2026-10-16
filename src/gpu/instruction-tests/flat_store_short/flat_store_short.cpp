/*
 * HIP test for FLAT_STORE_SHORT (ENC_FLAT, 16-bit flat store).
 *
 *   MEM[addr] = DATA[15:0]
 *
 * Each thread stores the low 16 bits of its value to a disjoint slot.
 * Pattern C, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o flat_store_short flat_store_short.cpp
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
kernel(uint16_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t val = (uint32_t)(0xABCD + i*7);
    uint64_t addr = (uint64_t)(out + i);
    __asm__ volatile(
        "flat_store_short %0, %1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : : "v"(addr), "v"(val) : "memory");
}

int main(void)
{
    uint16_t *h_out=(uint16_t*)malloc(N*sizeof(uint16_t));
    uint16_t *d_out; GPU_CHECK(hipMalloc(&d_out,N*sizeof(uint16_t)));
    GPU_CHECK(hipMemset(d_out,0,N*sizeof(uint16_t)));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(uint16_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint16_t exp=(uint16_t)(0xABCD+i*7); if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%04x exp 0x%04x\n",i,h_out[i],exp);pass=0;break;} }
    if(pass)printf("out[0]=0x%04x out[%d]=0x%04x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
