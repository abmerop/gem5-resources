/*
 * HIP test for S_BFE_I64 (SOP2 signed bit-field extract, 64-bit).
 *
 *   off   = S1[5:0]      width = S1[22:16]
 *   D.i64 = sign_extend( (S0.i64 >> off) & ((1<<width)-1), width )
 *   SCC   = (D != 0)
 *
 * Matches gem5 sop2.cc exactly.  Widths kept <=20 and off+width<=64 so the
 * masked field is identical under arithmetic vs logical shift and the host
 * (1<<width) reference has no UB.
 *
 * Pattern B (scalar): operands are kernel arguments (already in SGPRs); one
 * wavefront per block computes one result; lane 0 writes it.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_bfe_i64 s_bfe_i64.cpp
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
kernel_s_bfe_i64(uint64_t s0, uint32_t s1, uint64_t *out)
{
    uint64_t sdst;
    __asm__ volatile("s_bfe_i64 %0, %1, %2\n\t"
                     : "=s"(sdst) : "s"(s0), "s"(s1));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    struct { uint64_t s0; uint32_t s1; } cases[] = {
        {0x123456789abcdef0ULL, (0u)|(16u<<16)},   /* low 16 = 0xdef0 (neg) */
        {0x123456789abcdef0ULL, (16u)|(16u<<16)},  /* 0x9abc (neg) */
        {0x123456789abcdef0ULL, (32u)|(16u<<16)},  /* 0x5678 (pos) */
        {0x0000000000000040ULL, (0u)|(8u<<16)},    /* 0x40 (pos) */
        {0x0000000000000080ULL, (0u)|(8u<<16)},    /* 0x80 -> sign ext */
        {0xffffffffffffffffULL, (8u)|(4u<<16)},    /* all ones field */
        {0x00000000000abc00ULL, (8u)|(12u<<16)},   /* 0xabc (neg, 12-bit) */
        {0x7fffffffffffffffULL, (40u)|(20u<<16)},
    };
    const int nc = (int)(sizeof(cases)/sizeof(cases[0]));
    uint64_t *d_out=nullptr;
    GPU_CHECK(hipMalloc(&d_out,nc*sizeof(uint64_t)));
    for(int c=0;c<nc;c++){
        kernel_s_bfe_i64<<<1,BLOCK_SIZE>>>(cases[c].s0,cases[c].s1,d_out+c);
        GPU_CHECK(hipGetLastError());
    }
    GPU_CHECK(hipDeviceSynchronize());
    uint64_t *h_out=(uint64_t*)malloc(nc*sizeof(uint64_t));
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*sizeof(uint64_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<nc;c++){
        uint64_t s0=cases[c].s0; uint32_t s1=cases[c].s1;
        uint32_t off=s1&0x3fu, width=(s1>>16)&0x7fu;
        uint64_t mask=((uint64_t)1<<width)-1ULL;
        uint64_t field=(s0>>off)&mask;
        int64_t exp=(int64_t)field;
        if(width && ((field>>(width-1))&1ULL)) exp = (int64_t)(field | (~0ULL<<width));
        if((uint64_t)h_out[c]!=(uint64_t)exp){
            fprintf(stderr,"FAIL case %d: got 0x%llx exp 0x%llx\n",c,
                (unsigned long long)h_out[c],(unsigned long long)exp);pass=0;break;
        }
    }
    if(pass)printf("out[0]=0x%llx\n",(unsigned long long)h_out[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
