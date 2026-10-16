/*
 * HIP test for S_NOT_B64 (ENC_SOP1, 64-bit bitwise NOT, sets SCC).
 *
 *   SDST = ~S0 ; SCC = (SDST != 0)
 *
 * Pattern B (scalar): operands are kernel arguments; lane 0 writes the result.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_not_b64 s_not_b64.cpp
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
kernel(uint64_t s0, uint64_t *out)
{
    uint64_t sdst;
    __asm__ volatile("s_not_b64 %0, %1\n\t" : "=s"(sdst) : "s"(s0));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    uint64_t cases[] = {
        0x0000000000000000ULL, 0xffffffffffffffffULL, 0x0123456789abcdefULL,
        0xdeadbeefcafef00dULL, 0x8000000000000001ULL, 0x00000000ffffffffULL,
    };
    const int nc=(int)(sizeof(cases)/sizeof(cases[0]));
    uint64_t *d_out; GPU_CHECK(hipMalloc(&d_out,nc*sizeof(uint64_t)));
    for(int c=0;c<nc;c++){ kernel<<<1,BLOCK_SIZE>>>(cases[c],d_out+c); GPU_CHECK(hipGetLastError()); }
    GPU_CHECK(hipDeviceSynchronize());
    uint64_t *h_out=(uint64_t*)malloc(nc*sizeof(uint64_t));
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*sizeof(uint64_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<nc;c++){
        uint64_t exp=~cases[c];
        if(h_out[c]!=exp){fprintf(stderr,"FAIL case %d got 0x%llx exp 0x%llx\n",c,(unsigned long long)h_out[c],(unsigned long long)exp);pass=0;break;}
    }
    if(pass)printf("out[0]=0x%llx\n",(unsigned long long)h_out[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
