/*
 * HIP test for S_DECPERFLEVEL (ENC_SOPP, decrease performance level hint).
 * Pattern B.
 *
 * S_DECPERFLEVEL takes a 16-bit immediate performance level and, on hardware,
 * is a scheduling/perf hint with no effect on program-visible state.  A
 * correct functional model should treat it as a NOP, so this test embeds
 * `s_decperflevel 0` in the middle of a simple element-wise add kernel and
 * verifies the arithmetic result is unaffected.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950): s_decperflevel 0
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_decperflevel s_decperflevel.cpp
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

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kernel(const int *a, const int *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int s = a[i] + b[i];
    __asm__ volatile("s_decperflevel 0\n\t" ::: "memory");
    out[i] = s;
}

int
main(void)
{
    int *h_a = (int *)malloc(N * sizeof(int));
    int *h_b = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i + 1;
    }
    int *d_a, *d_b, *d_out;
    GPU_CHECK(hipMalloc(&d_a, N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_b, N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(int), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(int), hipMemcpyHostToDevice));
    kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_a[i] + h_b[i]) {
            fprintf(stderr, "FAIL[%d] got %d exp %d\n", i, h_out[i],
                    h_a[i] + h_b[i]);
            pass = 0;
            break;
        }
    }
    if (pass)
        printf("out[0]=%d out[%d]=%d\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
