/*
 * HIP test for V_CVT_SR_BF16_F32 (VOP3, stochastic-rounding f32 -> bf16).
 *
 * gem5 implements this with convertMXFP(..., roundStochastic, src1).  To make
 * the result deterministic and independent of the stochastic value, every
 * input f32 is chosen to be EXACTLY representable in bf16 (its low 16 mantissa
 * bits are zero).  With no bits dropped, stochastic rounding never rounds up,
 * so:
 *
 *   VDST[15:0]  = bf16(src0) = (src0_bits >> 16)      (== the chosen pattern)
 *   VDST[31:16] = preserved
 *
 * src1 (the stochastic value) is varied per lane to prove independence.  Only
 * the low 16 bits of the destination are checked; base VOP3 form only (no
 * DPP/SDWA -- gem5 panics on those).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_sr_bf16_f32 v_cvt_sr_bf16_f32.cpp
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
                    hipGetErrorString(e));                                   \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N = 256; static const int BLOCK_SIZE = 64;

__global__ void
kbase(const uint32_t *src, const uint32_t *sr, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint32_t s0 = src[i], s1 = sr[i];
    uint32_t vd = 0xABCD0000u;       // preserved high bits marker
    __asm__ volatile("v_cvt_sr_bf16_f32 %0, %1, %2\n\t"
                     : "+v"(vd) : "v"(s0), "v"(s1));
    out[i] = vd;
}

int main(void) {
    uint32_t *h_src = (uint32_t*)malloc(N*4);
    uint32_t *h_sr  = (uint32_t*)malloc(N*4);
    uint32_t *h_bf  = (uint32_t*)malloc(N*4);  // expected bf16 pattern
    uint32_t *h_out = (uint32_t*)malloc(N*4);
    for (int i = 0; i < N; i++) {
        uint32_t gen = 0x9e3779b9u * (uint32_t)(i + 1);
        // Build a finite normal bf16 pattern: random sign+mantissa, exp in
        // [20,219] so it is never inf/nan.
        uint16_t sign = (uint16_t)((gen >> 16) & 0x8000u);
        uint16_t mant = (uint16_t)(gen & 0x7fu);
        uint16_t exp  = (uint16_t)(((i % 200) + 20) << 7);
        uint16_t h = sign | exp | mant;
        h_bf[i]  = h;
        h_src[i] = ((uint32_t)h) << 16;          // f32 with zero low mantissa
        h_sr[i]  = gen ^ 0xdeadbeefu;            // varied stochastic value
    }
    uint32_t *d_src, *d_sr, *d_out;
    GPU_CHECK(hipMalloc(&d_src, N*4)); GPU_CHECK(hipMalloc(&d_sr, N*4));
    GPU_CHECK(hipMalloc(&d_out, N*4));
    GPU_CHECK(hipMemcpy(d_src, h_src, N*4, hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_sr, h_sr, N*4, hipMemcpyHostToDevice));
    int g = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kbase, dim3(g), dim3(BLOCK_SIZE), 0, 0, d_src, d_sr, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N*4, hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint16_t e = (uint16_t)h_bf[i];
        uint16_t got = (uint16_t)(h_out[i] & 0xffffu);
        if (got != e) {
            fprintf(stderr, "FAIL[%d] src=0x%08x got low16 0x%04x exp 0x%04x\n",
                    i, h_src[i], got, e);
            pass = 0; break;
        }
    }
    if (pass) printf("out[1]=0x%08x out[5]=0x%08x\n", h_out[1], h_out[5]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
