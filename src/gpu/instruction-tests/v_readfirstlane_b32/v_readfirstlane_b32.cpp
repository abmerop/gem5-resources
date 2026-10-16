/*
 * HIP test for V_READFIRSTLANE_B32 (VOP1, broadcast first active lane to SGPR).
 * gem5 (vop1.cc): SDST = S0[findLsbSet(exec_mask)]. With a full exec mask the
 * first active lane is lane 0, so SDST = value in lane 0 of the source VGPR.
 * Each block is one wavefront (all 64 lanes active); the result equals the
 * source value from the block's lane 0 (a[blockIdx*64]).
 * Pattern A (vector src -> scalar dst). No DPP/SDWA.
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
kern(const uint32_t *a, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t v = a[i], sdst;
    __asm__ volatile("v_readfirstlane_b32 %0, %1\n\t" : "=s"(sdst) : "v"(v));
    /* All lanes receive the same scalar; lane 0 writes one result per block. */
    if (threadIdx.x == 0) out[blockIdx.x] = sdst;
}

int main(void)
{
    size_t sz = N * sizeof(uint32_t);
    uint32_t *h_a=(uint32_t*)malloc(sz);
    for (int i = 0; i < N; i++) h_a[i] = (uint32_t)(i * 2654435761u + 12345u);
    int grid = N / BLOCK_SIZE;
    uint32_t *d_a,*d_out;
    GPU_CHECK(hipMalloc(&d_a,sz));GPU_CHECK(hipMalloc(&d_out,grid*sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_a,h_a,sz,hipMemcpyHostToDevice));
    kern<<<grid,BLOCK_SIZE>>>(d_a,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint32_t h_out[8];
    GPU_CHECK(hipMemcpy(h_out,d_out,grid*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass = 1;
    for (int b = 0; b < grid; b++) {
        uint32_t e = h_a[b * BLOCK_SIZE];
        int ok = (h_out[b]==e);
        printf("  block %d -> 0x%08X exp 0x%08X %s\n", b, h_out[b], e, ok?"OK":"FAIL");
        if (!ok) pass = 0;
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_out));
    free(h_a);
    return pass ? 0 : 1;
}
