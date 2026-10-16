/*
 * HIP test for V_CVT_PK_F32_BF8 (VOP1, convert a packed pair of bf8 (e5m2)
 * bytes to two f32 values in a VGPR pair).
 *
 * gem5 (fp8_e5m2: 5 exp / 2 mant, bias 15, has Inf at 0x7c/0xfc, NaN when
 * exp==0x1f && mant!=0) reads a 16-bit word, converts its low byte -> dst.lo
 * and high byte -> dst.hi.  DPP panics in gem5 and the SDWA form aborts gem5
 * (operand.hh "!scalar" assertion), so only the base e32 form (word0) is
 * exercised.
 *
 *   base : v_cvt_pk_f32_bf8_e32  v[lo:hi], vsrc                 word0
 *
 * bf8 -> f32 is exact for finite values; Inf/NaN checked with isinf/isnan.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_pk_f32_bf8 v_cvt_pk_f32_bf8.cpp
 */
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static float u2f(uint32_t u) { union { uint32_t u; float f; } c; c.u = u; return c.f; }

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
kbase(const uint32_t *src, uint64_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint32_t s = src[i]; uint64_t vd;
    __asm__ volatile("v_cvt_pk_f32_bf8_e32 %0, %1\n\t" : "=v"(vd) : "v"(s));
    out[i] = vd;
}
// Reference e5m2 decode.  cls: 0 finite, 1 +inf, 2 -inf, 3 nan.
static float e5m2_ref(uint8_t b, int *cls) {
    int s = (b >> 7) & 1, e = (b >> 2) & 0x1F, m = b & 0x3;
    *cls = 0;
    double v;
    if (e == 0) {                       // zero / subnormal: 2^-14 * m/4
        v = ldexp((double)m, -16);
    } else if (e == 0x1F) {
        if (m == 0) { *cls = s ? 2 : 1; return 0.0f; }   // +/-inf
        *cls = 3; return 0.0f;                            // nan
    } else {
        v = ldexp(1.0 + m / 4.0, e - 15);
    }
    return (float)(s ? -v : v);
}

static int check(const char *tag, int i, uint8_t b, float got) {
    int cls; float e = e5m2_ref(b, &cls);
    if (cls == 3) { if (!isnan(got)) { fprintf(stderr,"FAIL %s[%d] bf8=0x%02x got %g exp NaN\n",tag,i,b,got); return 0; } }
    else if (cls == 1) { if (!(isinf(got) && got > 0)) { fprintf(stderr,"FAIL %s[%d] bf8=0x%02x got %g exp +Inf\n",tag,i,b,got); return 0; } }
    else if (cls == 2) { if (!(isinf(got) && got < 0)) { fprintf(stderr,"FAIL %s[%d] bf8=0x%02x got %g exp -Inf\n",tag,i,b,got); return 0; } }
    else if (got != e) { fprintf(stderr,"FAIL %s[%d] bf8=0x%02x got %g exp %g\n",tag,i,b,got); return 0; }
    return 1;
}

int main(void) {
    uint32_t *h_src = (uint32_t*)malloc(N*4);
    uint64_t *h_base = (uint64_t*)malloc(N*8);
    for (int i = 0; i < N; i++) {
        uint8_t va = (uint8_t)i, vb = (uint8_t)((i * 7 + 3) & 0xff);
        // word0 = (vb<<8)|va
        h_src[i] = ((uint32_t)va) | ((uint32_t)vb << 8);
    }
    uint32_t *d_src; uint64_t *d_base;
    GPU_CHECK(hipMalloc(&d_src, N*4)); GPU_CHECK(hipMalloc(&d_base, N*8));
    GPU_CHECK(hipMemcpy(d_src, h_src, N*4, hipMemcpyHostToDevice));
    int g = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kbase, dim3(g), dim3(BLOCK_SIZE), 0, 0, d_src, d_base, N); GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base, d_base, N*8, hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        uint8_t va = (uint8_t)i, vb = (uint8_t)((i * 7 + 3) & 0xff);
        float blo = u2f((uint32_t)h_base[i]), bhi = u2f((uint32_t)(h_base[i] >> 32));
        pass = check("base.lo", i, va, blo) && check("base.hi", i, vb, bhi);
    }
    if (pass) printf("base.lo[0x38]=%g\n", u2f((uint32_t)h_base[0x38]));
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
