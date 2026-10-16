/*
 * HIP test for V_BFREV_B32 (ENC_VOP1, bit reverse).
 * gem5 (vop1.cc): D.u32 = reverseBits(S0.u32).
 *
 * NOTE on variants: gem5's execute() has panic_if(isDPPInst())/
 * panic_if(isSDWAInst()) guards, so the DPP/SDWA forms abort the simulator.
 * Only BASE is exercised (recorded variants=base).
 *
 * Pattern A (vector): per-lane "v" operand forces the VOP1 encoding.
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
kern(const uint32_t *a, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s0 = a[i], d;
    __asm__ volatile("v_bfrev_b32 %0, %1\n\t" : "=v"(d) : "v"(s0));
    out[i] = d;
}

static uint32_t rev32(uint32_t v){ uint32_t r=0; for(int b=0;b<32;b++){ r=(r<<1)|(v&1); v>>=1;} return r; }

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++) h_a[i]=(uint32_t)(i*2654435761u) ^ (uint32_t)(i<<11);
    uint32_t *d_a,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t exp=rev32(h_a[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%08x out[%d]=%08x\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_out));free(h_a);free(h_out);
    return pass?0:1;
}
