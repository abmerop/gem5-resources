/*
 * HIP test for S_CMPK_EQ_U32 (SOPK: SCC = (sdst.u32 == (u32)SIMM16)).
 * gem5 (sopk.cc): SCC = (src.u32 == (u32)simm16), simm16 zero-extended.
 * Fixed immediate (0x1234 = 4660).  Captured via S_CSELECT_B32.  Variants: base.
 * Pattern B (scalar).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

#define IMM 0x1234u   /* 4660 */

static const int BLOCK_SIZE = 64;

__global__ void
kern(uint32_t v, uint32_t *out)
{
    uint32_t sdst;
    __asm__ volatile("s_cmpk_eq_u32 %1, 0x1234\n\t"
                     "s_cselect_b32 %0, 1, 0\n\t"
                     : "=s"(sdst) : "s"(v));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    uint32_t cases[] = {
        0u, 0x1234u, 0x1233u, 0x1235u, 0xffffffffu, 4660u,
        0x00001234u, 0x12340000u, 1u, 0xdeadbeefu, 0x1234u, 100u,
    };
    const int NC = sizeof(cases)/sizeof(cases[0]);
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out, NC*4));
    for (int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c], d_out+c);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int c=0;c<NC;c++) {
        uint32_t exp = (cases[c] == IMM) ? 1u : 0u;
        if (h[c]!=exp){fprintf(stderr,"FAIL[%d] v=0x%x got %u exp %u\n",c,cases[c],h[c],exp);pass=0;}
    }
    if (pass) printf("out[1]=%u out[2]=%u\n", h[1], h[2]);
    printf("%s\n", pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
