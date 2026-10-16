/*
 * HIP test for V_CVT_F32_FP8 (VOP1, convert one fp8 (e4m3, OCP) byte to f32).
 *
 * gem5 (fp8_e4m3: 4 exp / 3 mant, bias 7, NaN only at 0x7f/0xff, no Inf)
 * converts the selected source byte to f32.  DPP is not implemented in gem5
 * (panics) and the SDWA form aborts gem5 (operand.hh "!scalar" assertion), so
 * only the base e32 form (byte 0) is exercised.
 *
 *   base : v_cvt_f32_fp8_e32  vd, vsrc                 -> f32(byte0)
 *
 * The conversion of any finite fp8 value to f32 is exact, so results are
 * checked exactly (as float, which treats +0/-0 equal); NaN inputs are checked
 * with isnan().
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_f32_fp8 v_cvt_f32_fp8.cpp
 */
#include <hip/hip_runtime.h>
#include <math.h>
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

static const int N = 256; static const int BLOCK_SIZE = 64;

__global__ void
kbase(const uint32_t *src, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint32_t s = src[i]; float vd;
    __asm__ volatile("v_cvt_f32_fp8_e32 %0, %1\n\t" : "=v"(vd) : "v"(s));
    out[i] = vd;
}
// Reference e4m3 (OCP) decode; is_nan set for 0x7f/0xff.
static float e4m3_ref(uint8_t b, int *is_nan) {
    int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    *is_nan = 0;
    double v;
    if (e == 0) {                       // zero / subnormal: 2^-6 * m/8
        v = ldexp((double)m, -9);
    } else if (e == 0xF && m == 0x7) {  // NaN
        *is_nan = 1; return 0.0f;
    } else {                            // normal
        v = ldexp(1.0 + m / 8.0, e - 7);
    }
    return (float)(s ? -v : v);
}

int main(void) {
    uint32_t *h_src = (uint32_t*)malloc(N*4);
    float *h_base = (float*)malloc(N*4);
    for (int i = 0; i < N; i++) {
        uint8_t v = (uint8_t)i;             // byte0 = fp8 value
        h_src[i] = ((uint32_t)v);
    }
    uint32_t *d_src; float *d_base;
    GPU_CHECK(hipMalloc(&d_src, N*4)); GPU_CHECK(hipMalloc(&d_base, N*4));
    GPU_CHECK(hipMemcpy(d_src, h_src, N*4, hipMemcpyHostToDevice));
    int g = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kbase, dim3(g), dim3(BLOCK_SIZE), 0, 0, d_src, d_base, N); GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base, d_base, N*4, hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        int nan; float e = e4m3_ref((uint8_t)i, &nan);
        if (nan) {
            if (!isnan(h_base[i])) { fprintf(stderr,"FAIL base[%d] fp8=0x%02x got %g exp NaN\n",i,i,h_base[i]); pass=0; }
        } else {
            if (h_base[i] != e) { fprintf(stderr,"FAIL base[%d] fp8=0x%02x got %g exp %g\n",i,i,h_base[i],e); pass=0; }
        }
    }
    if (pass) printf("base[0x38]=%g base[0x40]=%g\n", h_base[0x38], h_base[0x40]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
