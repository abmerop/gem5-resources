/*
 * HIP test for S_BFM_B64 (SOP2 bit field mask, 64-bit).
 * gem5 (sop2.cc): sdst = ((1ULL << (src0[5:0])) - 1) << (src1[5:0]).
 *   src0 = field width, src1 = field offset.  Variants: base.
 * Pattern B (scalar): width/offset passed as kernel args (in SGPRs).
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
kern(uint32_t width, uint32_t off, uint64_t *out)
{
    uint64_t sdst;
    __asm__ volatile("s_bfm_b64 %0, %1, %2\n\t"
                     : "=s"(sdst) : "s"(width), "s"(off));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    struct { uint32_t w, o; } cases[] = {
        {0,0}, {1,0}, {4,0}, {8,4}, {16,8}, {32,0}, {32,16},
        {63,0}, {1,63}, {0,10}, {5,60}, {12,20},
    };
    const int NC = sizeof(cases)/sizeof(cases[0]);
    uint64_t *d_out; GPU_CHECK(hipMalloc(&d_out, NC*8));
    for (int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c].w, cases[c].o, d_out+c);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    uint64_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*8,hipMemcpyDeviceToHost));
    int pass=1;
    for (int c=0;c<NC;c++) {
        uint32_t w = cases[c].w & 63u, o = cases[c].o & 63u;
        uint64_t exp = ((1ULL << w) - 1ULL) << o;
        if (h[c]!=exp){fprintf(stderr,"FAIL[%d] got 0x%016llx exp 0x%016llx\n",
            c,(unsigned long long)h[c],(unsigned long long)exp);pass=0;}
    }
    if (pass) printf("out[0]=0x%016llx out[3]=0x%016llx\n",
                     (unsigned long long)h[0],(unsigned long long)h[3]);
    printf("%s\n", pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
