/*
 * HIP test for V_CMPX_LE_F16 (VOPC compare-and-set-EXEC, f16 less-or-equal).
 *
 * V_CMPX_LE_F16 sets, per lane, mask = (S0.f16 <= S1.f16), writes that mask to
 * the SGPR destination (VOP3 _e64 form) AND overwrites EXEC with it.  To keep
 * the rest of the wavefront usable, the kernel saves EXEC, runs the compare,
 * and restores EXEC (all scalar ops, which execute regardless of EXEC):
 *
 *   s_mov_b64 saved, exec
 *   v_cmpx_le_f16_e64 mask, va, vb      ; mask = exec = (va <= vb)
 *   s_mov_b64 exec, saved
 *
 * Each lane then extracts its own result bit from the mask.  Inputs are
 * exactly-representable f16 values (no NaN) so the ordered comparison is
 * unambiguous.
 *
 * gem5 status note: Inst_VOPC__V_CMPX_LE_F16::execute() is panicUnimplemented()
 * as of this writing, so this test is expected to abort the simulator.  It is
 * provided as a ready self-checking artifact once gem5 implements the op.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_cmpx_le_f16 v_cmpx_le_f16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_cmpx_le_f16
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
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

static const int N          = 256;
static const int BLOCK_SIZE = 64;       /* one wavefront per block */

__global__ void
kernel_v_cmpx_le_f16(const __half *a, const __half *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint16_t ha, hb;
    __builtin_memcpy(&ha, a + i, sizeof(ha));
    __builtin_memcpy(&hb, b + i, sizeof(hb));
    uint32_t va = ha, vb = hb;

    uint64_t mask, saved;
    __asm__ volatile(
        "s_mov_b64 %1, exec\n\t"
        "v_cmpx_le_f16_e64 %0, %2, %3\n\t"
        "s_mov_b64 exec, %1\n\t"
        : "=s"(mask), "=s"(saved)
        : "v"(va), "v"(vb));

    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int
main(void)
{
    __half *h_a = (__half *)malloc(N * sizeof(__half));
    __half *h_b = (__half *)malloc(N * sizeof(__half));
    int    *h_out = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_a[i] = __float2half((float)(i % 13) * 0.5f - 3.0f);
        h_b[i] = __float2half((float)(i % 7) * 0.5f - 1.5f);
    }

    __half *d_a, *d_b;
    int    *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(__half), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(__half), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cmpx_le_f16,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int expected = (__half2float(h_a[i]) <= __half2float(h_b[i])) ? 1 : 0;
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %d expected %d\n",
                    i, h_out[i], expected);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=%d out[%d]=%d\n", h_out[0], N - 1, h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);

    return pass ? 0 : 1;
}
