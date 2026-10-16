/*
 * HIP test for SCRATCH_LOAD_SBYTE (ENC_FLAT/scratch, 8-bit signed load from
 * private/scratch memory, sign-extended to 32 bits).  Pattern C, base only.
 *
 * gem5 decodes SCRATCH_LOAD_SBYTE onto the FLAT_LOAD_SBYTE class operating on
 * the scratch segment:
 *   VDST = sign_extend_8_to_32( SCRATCH8[flat_scratch + voffset] )
 *
 * Each lane performs a self-consistent round-trip in its own private scratch
 * space: scratch_store_byte writes an 8-bit value, then scratch_load_sbyte
 * reads it back sign-extended.  Negative payloads confirm the sign extension.
 * A dynamically-indexed volatile private array forces the kernel to reserve a
 * scratch (private) segment so flat_scratch is valid.  pad[] is only a liveness
 * device and is intentionally not read after the asm (the asm's scratch usage
 * overlaps pad[]'s bytes; see scratch_load_sshort for the same idiom).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o scratch_load_sbyte scratch_load_sbyte.cpp
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
kernel_scratch_load_sbyte(const int8_t *in, const int *idx, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    volatile int pad[8];
    int sel = idx[i] & 7;
    pad[sel] = i;

    int   val   = (int)in[i];   /* sign-extended payload to store (low byte) */
    int   voff  = 16;           /* per-lane scratch byte offset, inside the
                                 * compiler-reserved private region */
    int   result;

    __asm__ volatile(
        "scratch_store_byte %1, %2, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "scratch_load_sbyte %0, %1, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(result)
        : "v"(voff), "v"(val)
        : "memory");

    out[i] = result;
}

int
main(void)
{
    int8_t *h_in  = (int8_t *)malloc(N * sizeof(int8_t));
    int    *h_idx = (int    *)malloc(N * sizeof(int));
    int    *h_out = (int    *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        /* Mix of positive and negative bytes to exercise sign extension. */
        h_in[i]  = (int8_t)((i * 37 + 11) - 128);
        h_idx[i] = i;
    }

    int8_t *d_in;
    int    *d_idx, *d_out;
    GPU_CHECK(hipMalloc(&d_in,  N * sizeof(int8_t)));
    GPU_CHECK(hipMalloc(&d_idx, N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_in,  h_in,  N * sizeof(int8_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_idx, h_idx, N * sizeof(int),    hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_scratch_load_sbyte, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_in, d_idx, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int expected = (int)h_in[i];   /* sign-extended 8->32 */
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
