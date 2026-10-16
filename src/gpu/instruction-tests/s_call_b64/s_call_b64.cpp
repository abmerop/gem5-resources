/*
 * HIP test for S_CALL_B64 (ENC_SOPK, short-range scalar call).  Pattern B.
 *
 * S_CALL_B64 saves the return address (PC of the next instruction) into the
 * SGPR pair SDST and branches to PC + signed16*4.  A functional call to the
 * immediately-following label is architecturally a NOP for data (it just
 * records the return address and falls through), so this test embeds
 * `s_call_b64 s[dst], 1f` followed by the `1:` label inside a simple
 * element-wise add kernel and verifies the arithmetic result is unaffected.
 * The saved return address is written out to keep the call live.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950): s_call_b64 s[8:9], 1f
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_call_b64 s_call_b64.cpp
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
kernel(const int *a, const int *b, int *out, uint64_t *ret, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int s = a[i] + b[i];
    uint64_t r;
    /* Call to the next instruction: records return addr in %0, falls through. */
    __asm__ volatile("s_call_b64 %0, 1f\n\t"
                     "1:\n\t"
                     : "=s"(r)::"memory");
    out[i] = s;
    if (i == 0)
        *ret = r;
}

int
main(void)
{
    int *h_a = (int *)malloc(N * sizeof(int));
    int *h_b = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));
    uint64_t h_ret = 0;
    for (int i = 0; i < N; i++) {
        h_a[i] = 7 * i - 1;
        h_b[i] = 2 * i;
    }
    int *d_a, *d_b, *d_out;
    uint64_t *d_ret;
    GPU_CHECK(hipMalloc(&d_a, N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_b, N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_ret, sizeof(uint64_t)));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(int), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(int), hipMemcpyHostToDevice));
    kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_out, d_ret, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(&h_ret, d_ret, sizeof(uint64_t), hipMemcpyDeviceToHost));
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
        printf("out[0]=%d out[%d]=%d ret=0x%llx\n", h_out[0], N - 1,
               h_out[N - 1], (unsigned long long)h_ret);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
