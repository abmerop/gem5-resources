/*
 * HIP test for S_CMP_GT_U32 (SOPC unsigned 32-bit greater-than -> SCC).
 * gem5 (sopc.cc): SCC = (src0.u32 > src1.u32).  Captured via S_CSELECT_B32.
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
kern(uint32_t s0, uint32_t s1, uint32_t *out)
{
    uint32_t sdst;
    __asm__ volatile("s_cmp_gt_u32 %1, %2\n\t"
                     "s_cselect_b32 %0, 1, 0\n\t"
                     : "=s"(sdst) : "s"(s0), "s"(s1));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    struct { uint32_t a, b; } cases[] = {
        {0,0}, {1,0}, {0,1}, {5,5}, {0xffffffff,0}, {0,0xffffffff},
        {0x80000000,0x7fffffff}, {0x7fffffff,0x80000000}, {100,99}, {99,100},
        {0xdeadbeef,0xdeadbeef}, {0xffffffff,0xfffffffe},
    };
    const int NC = sizeof(cases)/sizeof(cases[0]);
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out, NC*4));
    for (int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c].a, cases[c].b, d_out+c);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int c=0;c<NC;c++) {
        uint32_t exp = (cases[c].a > cases[c].b) ? 1u : 0u;
        if (h[c]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",c,h[c],exp);pass=0;}
    }
    if (pass) printf("out[1]=%u out[4]=%u\n", h[1], h[4]);
    printf("%s\n", pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
