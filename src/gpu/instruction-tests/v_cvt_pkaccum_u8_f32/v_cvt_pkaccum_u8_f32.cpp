/*
 * HIP test for V_CVT_PKACCUM_U8_F32 (VOP3).
 *
 * Convert src0 (f32) to an unsigned 8-bit integer (truncation toward zero)
 * and store it into byte (src1 & 3) of the destination.  This opcode is
 * read-modify-write: it reads the prior destination value and only replaces
 * the selected byte (the ISA passes the destination in as src_c).  The
 * destination register is therefore initialized to 0 and declared "+v" so
 * hardware reads a known-zero prior value; the three unselected bytes stay 0:
 *
 *   out = (uint32_t)(uint8_t)(uint32_t)src0_f32  <<  8 * (src1 & 3)
 *
 * VOP3-only, no DPP/SDWA variants.  Inputs kept in [0,250] so the
 * float->uint conversion is well defined.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_pkaccum_u8_f32 v_cvt_pkaccum_u8_f32.cpp
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

static const int N = 256; static const int BLOCK_SIZE = 64;

__global__ void
kbase(const float *a, const uint32_t *sel, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float s0 = a[i]; uint32_t s1 = sel[i], vd = 0;
    __asm__ volatile("v_cvt_pkaccum_u8_f32 %0, %1, %2\n\t"
                     : "+v"(vd) : "v"(s0), "v"(s1));
    out[i] = vd;
}

static uint32_t OP(float s0, uint32_t s1) {
    uint8_t b = (uint8_t)(uint32_t)s0;
    return (uint32_t)b << (8u * (s1 & 3u));
}

int main(void) {
    float *h_a = (float*)malloc(N*4);
    uint32_t *h_sel = (uint32_t*)malloc(N*4);
    uint32_t *h_base = (uint32_t*)malloc(N*4);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i % 251);
        h_sel[i] = (uint32_t)(i & 3);
    }
    float *d_a; uint32_t *d_sel, *d_base;
    GPU_CHECK(hipMalloc(&d_a, N*4)); GPU_CHECK(hipMalloc(&d_sel, N*4));
    GPU_CHECK(hipMalloc(&d_base, N*4));
    GPU_CHECK(hipMemcpy(d_a, h_a, N*4, hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_sel, h_sel, N*4, hipMemcpyHostToDevice));
    int g = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kbase, dim3(g), dim3(BLOCK_SIZE), 0, 0, d_a, d_sel, d_base, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base, d_base, N*4, hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t e = OP(h_a[i], h_sel[i]);
        if (h_base[i] != e) {
            fprintf(stderr, "FAIL base[%d] got 0x%08x exp 0x%08x\n", i, h_base[i], e);
            pass = 0; break;
        }
    }
    if (pass) printf("base[1]=0x%08x base[2]=0x%08x\n", h_base[1], h_base[2]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
