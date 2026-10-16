/*
 * HIP test for V_RCP_F32 (VOP1 f32 reciprocal) with DPP and SDWA variants.
 *
 * gem5 computes vdst = 1.0 / src via vop1Helper (src/arch/amdgpu/common/insts/
 * vop1.cc), so the SDWA src/dst selects and DPP lane swizzle are applied to
 * src0 before the reciprocal.  Inputs are exact powers of two so 1/x is exact
 * in f32 and can be compared bit-exactly.
 *
 *   base : v_rcp_f32_e32 vd, va                 out = 1/a[i]
 *   dpp  : v_rcp_f32_dpp ... quad_perm:[1,0,3,2]
 *            src0 gathered from quadBase+perm[lane&3], perm={1,0,3,2}
 *            out = 1/a[swizzled_lane]
 *   sdwa : v_rcp_f32_sdwa ... dst_sel:DWORD src0_sel:DWORD
 *            full-dword select (identity) -> out = 1/a[i]
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_rcp_f32 v_rcp_f32.cpp
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                   \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kbase(const float *a, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float s0 = a[i], vd;
    __asm__ volatile("v_rcp_f32_e32 %0, %1\n\t" : "=v"(vd) : "v"(s0));
    out[i] = vd;
}
__global__ void
kdpp(const float *a, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float s0 = a[i], vd;
    __asm__ volatile("v_rcp_f32_dpp %0, %1 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t"
                     : "=v"(vd) : "v"(s0));
    out[i] = vd;
}
__global__ void
ksdwa(const float *a, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float s0 = a[i], vd;
    __asm__ volatile("v_rcp_f32_sdwa %0, %1 dst_sel:DWORD "
                     "dst_unused:UNUSED_PAD src0_sel:DWORD\n\t"
                     : "=v"(vd) : "v"(s0));
    out[i] = vd;
}

int
main(void)
{
    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_base = (float *)malloc(N * sizeof(float));
    float *h_dpp  = (float *)malloc(N * sizeof(float));
    float *h_sdwa = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
        h_a[i] = ldexpf(1.0f, (i % 9) - 4);   /* 2^-4 .. 2^4, 1/x exact */

    float *d_a, *d_base, *d_dpp, *d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,    N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_base, N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_dpp,  N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_sdwa, N * sizeof(float)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(float), hipMemcpyHostToDevice));

    int g = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kbase, dim3(g), dim3(BLOCK_SIZE), 0, 0, d_a, d_base, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kdpp,  dim3(g), dim3(BLOCK_SIZE), 0, 0, d_a, d_dpp,  N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(ksdwa, dim3(g), dim3(BLOCK_SIZE), 0, 0, d_a, d_sdwa, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_base, d_base, N * sizeof(float), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp,  d_dpp,  N * sizeof(float), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa, d_sdwa, N * sizeof(float), hipMemcpyDeviceToHost));

    static const int perm[4] = {1, 0, 3, 2};
    int pass = 1;
    for (int i = 0; i < N; i++) {
        float eb = 1.0f / h_a[i];
        if (h_base[i] != eb) {
            fprintf(stderr, "FAIL base[%d] got %g exp %g\n", i, h_base[i], eb);
            pass = 0; break;
        }
        int lane = i % BLOCK_SIZE, wb = i - lane, sl = (lane & ~3) + perm[lane & 3];
        float ed = 1.0f / h_a[wb + sl];
        if (h_dpp[i] != ed) {
            fprintf(stderr, "FAIL dpp[%d] got %g exp %g\n", i, h_dpp[i], ed);
            pass = 0; break;
        }
        float es = 1.0f / h_a[i];
        if (h_sdwa[i] != es) {
            fprintf(stderr, "FAIL sdwa[%d] got %g exp %g\n", i, h_sdwa[i], es);
            pass = 0; break;
        }
    }

    if (pass)
        printf("base[1]=%g dpp[1]=%g sdwa[1]=%g\n", h_base[1], h_dpp[1], h_sdwa[1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_base));
    GPU_CHECK(hipFree(d_dpp)); GPU_CHECK(hipFree(d_sdwa));
    free(h_a); free(h_base); free(h_dpp); free(h_sdwa);
    return pass ? 0 : 1;
}
