/*
 * HIP test for v_add_i32 (VOP3A signed 32-bit integer add).
 *
 * v_add_i32 is the VOP3A encoding of 32-bit integer addition:
 *
 *   VDST = SRC0.i32 + SRC1.i32
 *
 * It differs from the VOP2 instruction v_add_u32_e32 in that it uses the
 * 64-bit VOP3 encoding, which allows each source to independently be a
 * VGPR, SGPR, or inline constant with optional abs/neg modifiers.
 * The compiler emits v_add_u32_e32 (VOP2) for the common two-VGPR case
 * and promotes to v_add_i32 (VOP3) when it needs the extra source
 * flexibility.  This test uses inline assembly to force the VOP3 form
 * explicitly.
 *
 * Build:
 *   hipcc -O2 -o v_add_i32 v_add_i32.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_add_i32
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

static const int N          = 256;
static const int BLOCK_SIZE = 64;

/*
 * v_add_i32_kernel
 *
 * Each thread adds a[i] and b[i] using the VOP3A v_add_i32 instruction and
 * writes the result to out[i].  The inline assembly uses:
 *
 *   v_add_i32 vdst, vsrc0, vsrc1
 *
 * "=v" — output VGPR (destination)
 * "v"  — input VGPR (source operand)
 *
 * No s_waitcnt is needed: v_add_i32 is a pure ALU instruction with no
 * memory dependency.
 */
__global__ void
v_add_i32_kernel(const int *a, const int *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int va = a[i];
    int vb = b[i];
    int vd;

    __asm__ volatile(
        "v_add_i32 %0, %1, %2\n\t"
        : "=v"(vd)
        : "v"(va), "v"(vb));

    out[i] = vd;
}

int
main(void)
{
    /* ----- host buffers ----- */
    int *h_a   = (int *)malloc(N * sizeof(int));
    int *h_b   = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));

    /* Mix of positive and negative values to exercise signed addition. */
    for (int i = 0; i < N; i++) {
        h_a[i] = i - N / 2;          /* -128 .. 127 */
        h_b[i] = (i % 2 == 0) ? i : -i; /* alternating sign */
    }

    /* ----- device buffers ----- */
    int *d_a, *d_b, *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(int), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(int), hipMemcpyHostToDevice));

    /* ----- launch ----- */
    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(v_add_i32_kernel,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    /* ----- copy back ----- */
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    /* ----- verify ----- */
    int pass = 1;
    for (int i = 0; i < N; i++) {
        int expected = h_a[i] + h_b[i];
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %d expected %d "
                    "(a=%d b=%d)\n", i, h_out[i], expected, h_a[i], h_b[i]);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=%d out[%d]=%d\n", h_out[0], N - 1, h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    /* ----- cleanup ----- */
    GPU_CHECK(hipFree(d_a));
    GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_out));
    free(h_a);
    free(h_b);
    free(h_out);

    return pass ? 0 : 1;
}
