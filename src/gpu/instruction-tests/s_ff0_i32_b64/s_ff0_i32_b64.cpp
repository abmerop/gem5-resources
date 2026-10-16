/*
 * HIP test for S_FF0_I32_B64 (SOP1 find first zero bit, 64-bit source).
 * gem5 (sop1.cc via findFirstZero): sdst = index of lowest 0 bit of src.u64,
 *   or -1 if src is all ones.  Variants: base.
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
kern(uint64_t s0, int32_t *out)
{
    int32_t sdst;
    __asm__ volatile("s_ff0_i32_b64 %0, %1\n\t" : "=s"(sdst) : "s"(s0));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    uint64_t cases[] = {
        0x0000000000000000ull, 0xffffffffffffffffull, 0x0000000000000001ull,
        0x0000000000000003ull, 0x00000000ffffffffull, 0xfffffffffffffffeull,
        0x7fffffffffffffffull, 0x123456789abcdef0ull, 0xfffffffeffffffffull,
        0x8000000000000000ull,
    };
    const int NC = sizeof(cases)/sizeof(cases[0]);
    int32_t *d_out; GPU_CHECK(hipMalloc(&d_out, NC*4));
    for (int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c], d_out+c);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    int32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int c=0;c<NC;c++) {
        int32_t exp;
        if (cases[c] == 0xffffffffffffffffull) exp = -1;
        else { exp = 0; uint64_t v = cases[c];
               while (v & 1ull) { v >>= 1; exp++; } }
        if (h[c]!=exp){fprintf(stderr,"FAIL[%d] in 0x%016llx got %d exp %d\n",
            c,(unsigned long long)cases[c],h[c],exp);pass=0;}
    }
    if (pass) printf("out[0]=%d out[1]=%d\n", h[0], h[1]);
    printf("%s\n", pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
