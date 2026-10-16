/*
 * HIP test for V_SWAP_B32 (VOP1, swap two 32-bit VGPRs).
 * gem5 (vop1.cc): tmp=VDST; VDST=SRC; SRC=tmp (per active lane).
 * Both operands are read-write; exercised with two "+v" constraints.
 * Pattern A (vector). No DPP/SDWA (not supported).
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
kern(const uint32_t *a, const uint32_t *b, uint32_t *oa, uint32_t *ob, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x = a[i], y = b[i];
    /* v_swap_b32 swaps the contents of the two operand registers. */
    __asm__ volatile("v_swap_b32 %0, %1\n\t" : "+v"(x), "+v"(y));
    oa[i] = x;   /* should now hold original b[i] */
    ob[i] = y;   /* should now hold original a[i] */
}

int main(void)
{
    size_t sz = N * sizeof(uint32_t);
    uint32_t *h_a=(uint32_t*)malloc(sz),*h_b=(uint32_t*)malloc(sz),
             *h_oa=(uint32_t*)malloc(sz),*h_ob=(uint32_t*)malloc(sz);
    for (int i = 0; i < N; i++) {
        h_a[i] = (uint32_t)(i * 2654435761u + 1u);
        h_b[i] = (uint32_t)(i * 40503u + 0xABCD0000u);
    }
    uint32_t *d_a,*d_b,*d_oa,*d_ob;
    GPU_CHECK(hipMalloc(&d_a,sz));GPU_CHECK(hipMalloc(&d_b,sz));
    GPU_CHECK(hipMalloc(&d_oa,sz));GPU_CHECK(hipMalloc(&d_ob,sz));
    GPU_CHECK(hipMemcpy(d_a,h_a,sz,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,sz,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_oa,d_ob,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_oa,d_oa,sz,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_ob,d_ob,sz,hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (h_oa[i] != h_b[i] || h_ob[i] != h_a[i]) {
            fprintf(stderr,"FAIL[%d] oa %u (exp %u) ob %u (exp %u)\n",
                    i,h_oa[i],h_b[i],h_ob[i],h_a[i]);pass=0;break;
        }
    }
    if (pass) printf("swapped[0]: a=%u b=%u\n",h_oa[0],h_ob[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_oa));GPU_CHECK(hipFree(d_ob));
    free(h_a);free(h_b);free(h_oa);free(h_ob);
    return pass ? 0 : 1;
}
