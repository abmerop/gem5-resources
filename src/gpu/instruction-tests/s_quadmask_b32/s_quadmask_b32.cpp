/*
 * HIP test for S_QUADMASK_B32 (SOP1 reduce each nibble to one bit).
 * gem5 (sop1.cc via quadMask): for each 4-bit group k (k=0..7),
 *   sdst bit k = (src & (0xF << 4k)) != 0.  SCC = (sdst != 0).
 * Pattern B (scalar).  Variants: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int BLOCK_SIZE = 64;

__global__ void
kern(uint32_t s0, uint32_t *out)
{
    uint32_t sdst;
    __asm__ volatile("s_quadmask_b32 %0, %1\n\t" : "=s"(sdst) : "s"(s0));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    uint32_t cases[] = {
        0x00000000u, 0xffffffffu, 0x00000001u, 0x0000000fu, 0x000000f0u,
        0x12345678u, 0x80000000u, 0xf0f0f0f0u, 0x0f0f0f0fu, 0xa5a5a5a5u,
        0x10101010u, 0x00010002u,
    };
    const int NC = sizeof(cases)/sizeof(cases[0]);
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out, NC*4));
    for (int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c], d_out+c);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int c=0;c<NC;c++) {
        uint32_t exp = 0;
        for (int k=0;k<8;k++) if ((cases[c] >> (4*k)) & 0xFu) exp |= (1u<<k);
        if (h[c]!=exp){fprintf(stderr,"FAIL[%d] in 0x%08x got 0x%02x exp 0x%02x\n",
            c,cases[c],h[c],exp);pass=0;}
    }
    if (pass) printf("out[0]=0x%02x out[5]=0x%02x\n", h[0], h[5]);
    printf("%s\n", pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
