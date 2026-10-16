/*
 * HIP test for S_ADDK_I32 (SOPK: sdst = sdst + sext16(SIMM16)).
 * gem5 (sopk.cc): sdst = src.i32 + sext<16>(simm16), where src == sdst (the
 * destination is also the source).  A fixed immediate (12345) is used; the
 * input value varies.  Variants: base.
 * Pattern B (scalar): "+s" read-write SGPR; immediate is an asm literal.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

#define IMM 12345

static const int BLOCK_SIZE = 64;

__global__ void
kern(int32_t v, int32_t *out)
{
    __asm__ volatile("s_addk_i32 %0, 12345\n\t" : "+s"(v));
    if (threadIdx.x == 0) out[blockIdx.x] = v;
}

int main(void)
{
    int32_t cases[] = {
        0, 1, -1, 100, -100, 1000000, -1000000,
        2147483647, -2147483648, 55555, -55555, 12345,
    };
    const int NC = sizeof(cases)/sizeof(cases[0]);
    int32_t *d_out; GPU_CHECK(hipMalloc(&d_out, NC*4));
    for (int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c], d_out+c);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    int32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int c=0;c<NC;c++) {
        int32_t exp = (int32_t)((uint32_t)cases[c] + (uint32_t)IMM);
        if (h[c]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",c,h[c],exp);pass=0;}
    }
    if (pass) printf("out[0]=%d out[1]=%d\n", h[0], h[1]);
    printf("%s\n", pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
