/*
 * HIP test for S_BFM_B32 (SOP2 bit-field mask).
 *
 * Pattern B (scalar): operands are kernel arguments (already in SGPRs); one
 * wavefront per block computes one result; lane 0 writes it.  Matches gem5's
 * expression in src/arch/amdgpu/common/insts/.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_bfm_b32 s_bfm_b32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int BLOCK_SIZE=64;

__global__ void
kernel_s_bfm_b32(uint32_t s0, uint32_t s1, uint32_t *out) {
    uint32_t sdst;
    __asm__ volatile("s_bfm_b32 %0, %1, %2\n\t" : "=s"(sdst) : "s"(s0), "s"(s1));
    if (threadIdx.x==0) out[blockIdx.x]=sdst;
}

static uint32_t ref(uint32_t s0, uint32_t s1) { return (((1u << (s0 & 31u)) - 1u) << (s1 & 31u)); }

int main(void) {
    struct { uint32_t s0,s1; } cases[] = { {0x1u,0x0u},{0x4u,0x0u},{0x8u,0x4u},{0x10u,0x8u},{0x5u,0x14u},{0xcu,0x10u},{0x3u,0x1cu},{0xau,0x16u} };
    const int nc=(int)(sizeof(cases)/sizeof(cases[0]));
    uint32_t *d_out=nullptr; GPU_CHECK(hipMalloc(&d_out,nc*4));
    for (int c=0;c<nc;c++) { kernel_s_bfm_b32<<<1,BLOCK_SIZE>>>(cases[c].s0,cases[c].s1,d_out+c); GPU_CHECK(hipGetLastError()); }
    GPU_CHECK(hipDeviceSynchronize());
    uint32_t *h_out=(uint32_t*)malloc(nc*4);
    GPU_CHECK(hipMemcpy(h_out,d_out,nc*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int c=0;c<nc;c++) {
        uint32_t e=ref(cases[c].s0,cases[c].s1);
        if (h_out[c]!=e) { fprintf(stderr,"FAIL case %d: got 0x%08x exp 0x%08x\n",c,h_out[c],e); pass=0; break; }
    }
    if (pass) printf("out[0]=0x%08x\n",h_out[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
