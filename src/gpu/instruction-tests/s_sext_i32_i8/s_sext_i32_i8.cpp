/*
 * HIP test for S_SEXT_I32_I8 (SOP1, sign-extend 8-bit to 32-bit scalar).
 * gem5 (sop1.cc): SDST = sext<8>(S0[7:0]).
 * Pattern B (scalar): kernel-argument scalar operand (SGPR). No SCC.
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
kern(uint32_t s0, int32_t *out)
{
    int32_t sdst;
    __asm__ volatile("s_sext_i32_i8 %0, %1\n\t" : "=s"(sdst) : "s"(s0));
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    uint32_t inputs[] = {
        0x00000000u, 0x00000001u, 0x0000007Fu, 0x00000080u, 0x000000FFu,
        0xFFFFFF80u, 0xDEADBE7Fu, 0xDEADBEFFu, 0x000000ABu, 0x12345678u,
    };
    const int nc = (int)(sizeof(inputs)/sizeof(inputs[0]));
    int32_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, nc*sizeof(int32_t)));
    for (int c = 0; c < nc; c++) kern<<<1,BLOCK_SIZE>>>(inputs[c], d_out+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    int32_t h_out[16];
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*sizeof(int32_t),hipMemcpyDeviceToHost));
    int pass = 1;
    for (int c = 0; c < nc; c++) {
        int32_t e = (int32_t)(int8_t)(inputs[c] & 0xFFu);
        int ok = (h_out[c]==e);
        printf("  0x%08X -> %d exp %d %s\n", inputs[c], h_out[c], e, ok?"OK":"FAIL");
        if (!ok) pass = 0;
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
