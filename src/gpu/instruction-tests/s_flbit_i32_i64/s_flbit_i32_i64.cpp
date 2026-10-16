/*
 * HIP test for S_FLBIT_I32_I64 (SOP1 find first bit opposite of sign, 64-bit).
 * gem5 (sop1.cc via firstOppositeSignBit): counting from the MSB (bit 63),
 *   return the 0-based index of the first bit that differs from the sign bit.
 *   Returns -1 if src is 0 or -1.  Variants: base.
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

static const int BLOCK_SIZE = 64;

__global__ void
kern(int64_t s0, int32_t *out)
{
    int32_t sdst;
    __asm__ volatile("s_flbit_i32_i64 %0, %1\n\t" : "=s"(sdst) : "s"(s0));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    int64_t cases[] = {
        0, -1, 1, 2, -2,
        (int64_t)0x4000000000000000ll,   /* +, first 1 at index 1 */
        (int64_t)0x0000000100000000ll,
        (int64_t)0x8000000000000000ll,   /* INT64_MIN, sign=1, first 0 at idx1 */
        (int64_t)0xffffffff00000000ll,
        (int64_t)0x7fffffffffffffffll,
        (int64_t)0xc000000000000000ll,
        123456789ll,
    };
    const int NC = sizeof(cases)/sizeof(cases[0]);
    int32_t *d_out; GPU_CHECK(hipMalloc(&d_out, NC*4));
    for (int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c], d_out+c);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    int32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int c=0;c<NC;c++) {
        int32_t exp;
        uint64_t v = (uint64_t)cases[c];
        if (cases[c]==0 || cases[c]==-1) exp = -1;
        else {
            int sign = (int)((v >> 63) & 1ull);
            int count = 0;
            for (int i=0;i<64;i++) {
                int bit = (int)((v >> (63 - i)) & 1ull);
                if (bit != sign) break;
                count++;
            }
            exp = count;
        }
        if (h[c]!=exp){fprintf(stderr,"FAIL[%d] in 0x%016llx got %d exp %d\n",
            c,(unsigned long long)v,h[c],exp);pass=0;}
    }
    if (pass) printf("out[2]=%d out[5]=%d\n", h[2], h[5]);
    printf("%s\n", pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
