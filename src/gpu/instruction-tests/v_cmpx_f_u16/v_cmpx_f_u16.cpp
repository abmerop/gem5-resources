/*
 * HIP test for V_CMPX_F_U16 (v_cmpx_f_u16_e64), CDNA4 / gfx950.
 *
 * Self-checking single-instruction test. The kernel issues the CMPX via
 * inline asm into an explicit SGPR-pair destination; each lane extracts its
 * own result bit. Host reference mirrors gem5's exact predicate: 0.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

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

__global__ void
kernel_test(const uint32_t *a, const uint32_t *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = a[i] & 0xffffu;
    uint32_t vb = b[i] & 0xffffu;
    uint64_t mask, saved;
    __asm__ volatile(
        "s_mov_b64 %1, exec\n\t"
        "v_cmpx_f_u16_e64 %0, %2, %3\n\t"
        "s_mov_b64 exec, %1\n\t"
        : "=s"(mask), "=s"(saved)
        : "v"(va), "v"(vb));

    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int
main(void)
{
    uint32_t *h_a = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_b = (uint32_t *)malloc(N * sizeof(uint32_t));
    int *h_out = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        switch (i % 6) {
        case 0: h_a[i] = 5;     h_b[i] = 5;     break;
        case 1: h_a[i] = 10;    h_b[i] = 4000;  break;
        case 2: h_a[i] = 4000;  h_b[i] = 10;    break;
        case 3: h_a[i] = 0;     h_b[i] = 65535; break;
        case 4: h_a[i] = 65535; h_b[i] = 0;     break;
        case 5: h_a[i] = 123;   h_b[i] = 123;   break;
        }
    }

    uint32_t *d_a, *d_b;
    int *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_test, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint16_t av = (uint16_t)h_a[i]; uint16_t bv = (uint16_t)h_b[i];
        int exp = (0) ? 1 : 0;
        if (h_out[i] != exp) {
            fprintf(stderr, "FAIL out[%d] case %d: got %d expected %d\n",
                    i, i % 6, h_out[i], exp);
            pass = 0; break;
        }
    }
    if (pass)
        printf("out[0..5]=%d%d%d%d%d%d\n",
               h_out[0], h_out[1], h_out[2], h_out[3], h_out[4], h_out[5]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);
    return pass ? 0 : 1;
}
