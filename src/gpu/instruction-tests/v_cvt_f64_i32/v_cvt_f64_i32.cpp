/*
 * HIP test for V_CVT_F64_I32 (ENC_VOP1, signed int32 -> double conversion).
 *
 * V_CVT_F64_I32 computes, per lane:
 *
 *   VDST.f64 = (double)SRC0.i32
 *
 * The 32-bit signed source is read from a VGPR; the 64-bit result occupies a
 * VGPR pair, bound here with "=v" on a C++ double.
 *
 * DPP variant note: this instruction supports a DPP encoding, but because the
 * result is double-precision the assembler classifies it as "DP ALU", which
 * only accepts the row_newbcast DPP control (DPP_CTRL 0x150-0x15F).  gem5's
 * vega DPP implementation (dppInstImpl in
 * src/arch/amdgpu/vega/insts/inst_util.hh) only handles DPP_CTRL values up to
 * 0x143 (ROW_BCAST31) and panics ("instruction using reserved DPP_CTRL
 * value") on row_newbcast.  Exercising the DPP form would therefore crash the
 * simulator rather than validate anything, so this test covers the base
 * encoding only.  (The DPP variant can be added once gem5 gains row_newbcast
 * support.)
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_cvt_f64_i32 v_cvt_f64_i32.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cvt_f64_i32
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

static const int N          = 256;
static const int BLOCK_SIZE = 64;

/*
 * kernel_v_cvt_f64_i32
 *
 * Each lane converts a[i] (i32) to f64 via the base VOP1 encoding.
 */
__global__ void
kernel_v_cvt_f64_i32(const int32_t *a, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int32_t va = a[i];
    double vd;
    __asm__ volatile(
        "v_cvt_f64_i32 %0, %1\n\t"
        : "=v"(vd)
        : "v"(va));

    out[i] = vd;
}

int
main(void)
{
    int32_t *h_a   = (int32_t *)malloc(N * sizeof(int32_t));
    double  *h_out = (double *)malloc(N * sizeof(double));

    /* Wide range of signed magnitudes, exactly representable in f64. */
    for (int i = 0; i < N; i++) {
        int32_t v = (i - 128) * 16777217;   /* spread across i32 range */
        h_a[i] = v;
    }

    int32_t *d_a;
    double  *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(int32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(double)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(int32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cvt_f64_i32,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(double),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        double expected = (double)h_a[i];     /* exact: i32 -> f64 */
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %f expected %f (a=%d)\n",
                    i, h_out[i], expected, h_a[i]);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=%f out[%d]=%f\n", h_out[0], N - 1, h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a));
    GPU_CHECK(hipFree(d_out));
    free(h_a);
    free(h_out);

    return pass ? 0 : 1;
}
