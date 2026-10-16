/*
 * HIP test for V_MBCNT_HI_U32_B32 (VOP3, masked bit count, high 32 lanes).
 * gem5 (vop3.cc): threadMask = (1<<lane)-1;
 *   D.u32 = popcount(S0.u32 & threadMask[63:32]) + S1.u32
 *
 * With S0 = 0xFFFFFFFF and S1 = 1000 on every lane:
 *   D[lane] = popcount(maskHigh32(lane)) + 1000 = max(0,lane-32) + 1000
 * Pattern A (vector), BLOCK_SIZE=64 = one wavefront/block, threadIdx.x==lane.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const uint32_t *a, const uint32_t *b, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s0 = a[i], s1 = b[i], d;
    __asm__ volatile("v_mbcnt_hi_u32_b32 %0, %1, %2\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1));
    out[i] = d;
}

int main(void)
{
    size_t sz = N * sizeof(uint32_t);
    uint32_t *h_a=(uint32_t*)malloc(sz),*h_b=(uint32_t*)malloc(sz),*h_o=(uint32_t*)malloc(sz);
    for (int i = 0; i < N; i++) { h_a[i] = 0xFFFFFFFFu; h_b[i] = 1000u; }
    uint32_t *d_a,*d_b,*d_o;
    GPU_CHECK(hipMalloc(&d_a,sz));GPU_CHECK(hipMalloc(&d_b,sz));GPU_CHECK(hipMalloc(&d_o,sz));
    GPU_CHECK(hipMemcpy(d_a,h_a,sz,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,sz,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,sz,hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N; i++) {
        int lane = i % BLOCK_SIZE;
        uint32_t e = (uint32_t)(lane < 32 ? 0 : lane - 32) + 1000u;
        if (h_o[i] != e) { fprintf(stderr,"FAIL[%d] lane %d got %u exp %u\n",i,lane,h_o[i],e);pass=0;break; }
    }
    if (pass) printf("out[0]=%u out[32]=%u out[63]=%u\n",h_o[0],h_o[32],h_o[63]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_o));
    free(h_a);free(h_b);free(h_o);
    return pass ? 0 : 1;
}
