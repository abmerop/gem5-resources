/*
 * HIP test for SCRATCH_STORE_BYTE (ENC_FLAT_SCRATCH, 8-bit store to
 * private/scratch memory).
 *
 *   SCRATCH8[flat_scratch + voffset] = VDATA[7:0]
 *
 * Each lane performs a self-consistent round-trip in its own private scratch
 * space: scratch_store_byte writes the low byte of a value, then
 * scratch_load_ubyte reads it back zero-extended.  A dynamically-indexed
 * volatile private array forces the kernel to reserve a scratch (private)
 * segment so flat_scratch is valid.  As in scratch_load_sshort, pad[] is only a
 * liveness device and is NOT read back after the asm (the asm's scratch usage
 * overlaps pad[]'s bytes).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o scratch_store_byte scratch_store_byte.cpp
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

__global__ void
kernel_scratch_store_byte(const int *in, const int *idx, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    /* Force a private (scratch) segment to be reserved. */
    volatile int pad[8];
    int sel = idx[i] & 7;
    pad[sel] = i;

    int val    = in[i];   /* low byte is stored */
    int voff   = 16;      /* per-lane scratch byte offset inside private region */
    int result;

    __asm__ volatile(
        "scratch_store_byte %1, %2, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "scratch_load_ubyte %0, %1, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(result)
        : "v"(voff), "v"(val)
        : "memory");

    out[i] = result;   /* zero-extended byte just stored */
}

int
main(void)
{
    int *h_in  = (int *)malloc(N * sizeof(int));
    int *h_idx = (int *)malloc(N * sizeof(int));
    int *h_out = (int *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        h_in[i]  = i * 277 + 13;   /* arbitrary; only low byte matters */
        h_idx[i] = i;
    }

    int *d_in, *d_idx, *d_out;
    GPU_CHECK(hipMalloc(&d_in,  N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_idx, N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_in,  h_in,  N * sizeof(int), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_idx, h_idx, N * sizeof(int), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_scratch_store_byte, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_in, d_idx, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int expected = h_in[i] & 0xff;   /* stored low byte, zero-extended */
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %d expected %d\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=%d out[%d]=%d\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_in)); GPU_CHECK(hipFree(d_idx)); GPU_CHECK(hipFree(d_out));
    free(h_in); free(h_idx); free(h_out);
    return pass ? 0 : 1;
}
