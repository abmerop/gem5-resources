/*
 * HIP test for S_LSHL1_ADD_U32 (SOP2: sdst = (src0 << 1) + src1).
 * gem5 (sop2.cc): sdst = (src0.u32 << 1) + src1.u32.  Variants: base.
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
kern(uint32_t s0, uint32_t s1, uint32_t *out)
{
    uint32_t sdst;
    __asm__ volatile("s_lshl1_add_u32 %0, %1, %2\n\t"
                     : "=s"(sdst) : "s"(s0), "s"(s1));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    struct { uint32_t a, b; } cases[] = {
        {0,0}, {1,0}, {0,1}, {1,1}, {0x10,0x100}, {0xff,0xf},
        {0x10000000,0x12345678}, {0xffffffff,1}, {7,0x80000000},
        {0x0f0f0f0f,0x0a0a0a0a},
    };
    const int NC = sizeof(cases)/sizeof(cases[0]);
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out, NC*4));
    for (int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c].a, cases[c].b, d_out+c);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int c=0;c<NC;c++) {
        uint32_t exp = (cases[c].a << 1) + cases[c].b;
        if (h[c]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",c,h[c],exp);pass=0;}
    }
    if (pass) printf("out[0]=0x%08x out[1]=0x%08x\n", h[0], h[1]);
    printf("%s\n", pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
