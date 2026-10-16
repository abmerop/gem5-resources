/*
 * HIP test for V_MSAD_U8 (VOP3, masked sum of absolute differences, bytes).
 * gem5 (vop3.cc): for each of the 4 bytes b in S0/S1,
 *   acc = S2; if (S1.byte[b] != 0) acc += |S0.byte[b] - S1.byte[b]|;
 *   D.u32 = acc.
 * A zero reference byte in S1 masks (skips) that byte's contribution.
 * Pattern A (vector): three "v" inputs, one "=v" output. No DPP/SDWA.
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
kern(const uint32_t *a, const uint32_t *b, const uint32_t *c,
     uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s0 = a[i], s1 = b[i], s2 = c[i], d;
    __asm__ volatile("v_msad_u8 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

static uint32_t
ref(uint32_t s0, uint32_t s1, uint32_t s2)
{
    uint32_t acc = s2;
    for (int b = 0; b < 4; b++) {
        uint32_t x = (s0 >> (8*b)) & 0xFF;
        uint32_t y = (s1 >> (8*b)) & 0xFF;
        if (y != 0) acc += (x > y) ? (x - y) : (y - x);
    }
    return acc;
}

int main(void)
{
    size_t sz = N * sizeof(uint32_t);
    uint32_t *h_a=(uint32_t*)malloc(sz),*h_b=(uint32_t*)malloc(sz),
             *h_c=(uint32_t*)malloc(sz),*h_o=(uint32_t*)malloc(sz);
    for (int i = 0; i < N; i++) {
        h_a[i] = (uint32_t)(i * 2654435761u);
        h_b[i] = (uint32_t)(i * 40503u + 0x00010200u); /* some zero bytes */
        h_c[i] = (uint32_t)(i * 7u);
    }
    uint32_t *d_a,*d_b,*d_c,*d_o;
    GPU_CHECK(hipMalloc(&d_a,sz));GPU_CHECK(hipMalloc(&d_b,sz));
    GPU_CHECK(hipMalloc(&d_c,sz));GPU_CHECK(hipMalloc(&d_o,sz));
    GPU_CHECK(hipMemcpy(d_a,h_a,sz,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,sz,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,sz,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,sz,hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t e = ref(h_a[i],h_b[i],h_c[i]);
        if (h_o[i] != e) { fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,h_o[i],e);pass=0;break; }
    }
    if (pass) printf("out[1]=%u out[%d]=%u\n",h_o[1],N-1,h_o[N-1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_o));
    free(h_a);free(h_b);free(h_c);free(h_o);
    return pass ? 0 : 1;
}
