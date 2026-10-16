/*
 * HIP test for S_FF0_I32_B32 (SOP1 find first zero bit (LSB)).  Pattern B scalar, one source.
 * Matches gem5's expression.
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_ff0_i32_b32 s_ff0_i32_b32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int BLOCK_SIZE=64;
__global__ void
kernel_s_ff0_i32_b32(uint32_t s0, uint32_t *out) {
    uint32_t sdst;
    __asm__ volatile("s_ff0_i32_b32 %0, %1\n\t" : "=s"(sdst) : "s"(s0));
    if (threadIdx.x==0) out[blockIdx.x]=sdst;
}
static uint32_t ref(uint32_t s0) { return (s0==0xffffffffu ? 0xffffffffu : (uint32_t)__builtin_ctz(~s0)); }
int main(void) {
    uint32_t cases[] = { 0x0u,0xffffffffu,0xfffffffeu,0x1u,0xdeadbeefu,0xaaaaaaaau,0x0u,0x7fffffffu };
    const int nc=(int)(sizeof(cases)/sizeof(cases[0]));
    uint32_t *d_out=nullptr; GPU_CHECK(hipMalloc(&d_out,nc*4));
    for (int c=0;c<nc;c++) { kernel_s_ff0_i32_b32<<<1,BLOCK_SIZE>>>(cases[c],d_out+c); GPU_CHECK(hipGetLastError()); }
    GPU_CHECK(hipDeviceSynchronize());
    uint32_t *h_out=(uint32_t*)malloc(nc*4);
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int c=0;c<nc;c++) {
        uint32_t e=ref(cases[c]);
        if (h_out[c]!=e) { fprintf(stderr,"FAIL case %d in=0x%08x got 0x%08x exp 0x%08x\n",c,cases[c],h_out[c],e); pass=0; break; }
    }
    if (pass) printf("out[0]=0x%08x\n",h_out[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
