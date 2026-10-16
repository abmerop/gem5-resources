/*
 * HIP test for S_SUBB_U32 (SOP2 scalar subtract with borrow-in).
 *
 * gem5 (sop2.cc): SDST = SSRC0 - SSRC1 - SCC; SCC = (SSRC1 + SCC) > SSRC0.
 * The instruction is exercised as the high half of a 64-bit scalar subtract:
 *     s_sub_u32  d_lo, a_lo, b_lo     ; SCC = borrow (b_lo > a_lo)
 *     s_subb_u32 d_hi, a_hi, b_hi     ; d_hi = a_hi - b_hi - SCC
 * so the 64-bit result equals a - b (mod 2^64), which the host verifies for a
 * set of operand pairs chosen to exercise borrow-in = 0 and 1.  Operands are
 * uniform kernel arguments, so the compiler keeps them in SGPRs (scalar).
 * Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

__device__ __forceinline__ uint64_t sub64(uint64_t a,uint64_t b){
    uint32_t a_lo=(uint32_t)a, a_hi=(uint32_t)(a>>32);
    uint32_t b_lo=(uint32_t)b, b_hi=(uint32_t)(b>>32);
    uint32_t d_lo,d_hi;
    __asm__ volatile(
        "s_sub_u32 %0, %2, %3\n\t"
        "s_subb_u32 %1, %4, %5\n\t"
        :"=s"(d_lo),"=s"(d_hi)
        :"s"(a_lo),"s"(b_lo),"s"(a_hi),"s"(b_hi)
        :"scc");
    return ((uint64_t)d_hi<<32)|d_lo;
}

__global__ void
kern(uint64_t a0,uint64_t b0,uint64_t a1,uint64_t b1,
     uint64_t a2,uint64_t b2,uint64_t a3,uint64_t b3,uint64_t*out){
    if(threadIdx.x!=0) return;
    out[0]=sub64(a0,b0); out[1]=sub64(a1,b1);
    out[2]=sub64(a2,b2); out[3]=sub64(a3,b3);
}
int main(void){
    uint64_t a[4]={0x0000000100000000ULL, 5ULL,
                   0xFFFFFFFFFFFFFFFFULL, 0x123456789ABCDEF0ULL};
    uint64_t b[4]={0x0000000000000001ULL, 10ULL,
                   1ULL, 0x0FEDCBA987654321ULL};
    uint64_t *d_o, ho[4];
    GPU_CHECK(hipMalloc(&d_o,4*8));
    kern<<<1,64>>>(a[0],b[0],a[1],b[1],a[2],b[2],a[3],b[3],d_o);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,d_o,4*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<4;i++){ uint64_t e=a[i]-b[i];
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %016llx exp %016llx\n",i,
            (unsigned long long)ho[i],(unsigned long long)e);pass=0;} }
    if(pass)printf("d0=%016llx d1=%016llx d2=%016llx d3=%016llx\n",
        (unsigned long long)ho[0],(unsigned long long)ho[1],
        (unsigned long long)ho[2],(unsigned long long)ho[3]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
