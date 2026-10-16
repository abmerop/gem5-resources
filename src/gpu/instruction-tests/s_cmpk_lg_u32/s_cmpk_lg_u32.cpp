/*
 * HIP test for S_CMPK_LG_U32 (ENC_SOPK, scalar compare-not-equal with imm16).
 *
 *   SCC = (S0.u32 != SIMM16)     (SIMM16 zero-extended to 32 bits)
 *
 * gem5 sopk.cc compares src against the 16-bit immediate.  We read SCC via
 * s_cselect_b32 (1 if SCC else 0).  Pattern B (scalar); the immediate is a
 * compile-time constant, so a separate kernel is used per immediate.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_cmpk_lg_u32 s_cmpk_lg_u32.cpp
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

/* SIMM16 = 0x1234 fixed; vary S0 across cases. */
__global__ void
kernel(uint32_t s0, uint32_t *out)
{
    uint32_t scc;
    __asm__ volatile(
        "s_cmpk_lg_u32 %1, 0x1234\n\t"
        "s_cselect_b32 %0, 1, 0\n\t"
        : "=s"(scc) : "s"(s0));
    if (threadIdx.x == 0) out[blockIdx.x] = scc;
}

int main(void)
{
    uint32_t cases[] = { 0x1234u, 0x1233u, 0x1235u, 0u, 0xffffffffu, 0x00001234u, 0x12340000u };
    const int nc=(int)(sizeof(cases)/sizeof(cases[0]));
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out,nc*sizeof(uint32_t)));
    for(int c=0;c<nc;c++){ kernel<<<1,BLOCK_SIZE>>>(cases[c],d_out+c); GPU_CHECK(hipGetLastError()); }
    GPU_CHECK(hipDeviceSynchronize());
    uint32_t *h_out=(uint32_t*)malloc(nc*sizeof(uint32_t));
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<nc;c++){ uint32_t exp=(cases[c]!=0x1234u)?1u:0u; if(h_out[c]!=exp){fprintf(stderr,"FAIL case %d (0x%x) got %u exp %u\n",c,cases[c],h_out[c],exp);pass=0;break;} }
    if(pass)printf("out[0]=%u out[1]=%u\n",h_out[0],h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
