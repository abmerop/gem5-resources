/*
 * HIP test for SCRATCH_LOAD_SSHORT (ENC_FLAT/scratch, 16-bit signed load from
 * private/scratch memory, sign-extended to 32 bits).
 *
 * gem5 decodes SCRATCH_LOAD_SSHORT as Inst_FLAT__FLAT_LOAD_SSHORT operating on
 * the scratch segment (src/arch/amdgpu/vega/gpu_decoder.cc):
 *
 *   VDST = sign_extend_16_to_32( SCRATCH16[flat_scratch + voffset] )
 *
 * Each lane performs a self-consistent round-trip in its own private scratch
 * space: scratch_store_short writes a 16-bit value, then scratch_load_sshort
 * reads it back sign-extended.  Negative payloads confirm the sign extension.
 * A dynamically-indexed volatile private array forces the kernel to reserve a
 * scratch (private) segment so flat_scratch is valid.
 *
 * NOTE: the inline-asm scratch access must stay inside the private region the
 * compiler reserves for pad[] (the asm's scratch usage is invisible to the
 * compiler, so it cannot reserve extra space).  That means the scratch
 * round-trip overlaps pad[]'s bytes, so pad[] must NOT be read back after the
 * asm -- doing so reads bytes the round-trip clobbered.  We therefore verify
 * the self-contained 'result' directly and use pad[] only as a liveness device
 * to force the scratch allocation.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o scratch_load_sshort scratch_load_sshort.cpp
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
kernel_scratch_load_sshort(const int16_t *in, const int *idx, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    /* Force the kernel to reserve a private (scratch) segment.  This volatile,
     * dynamically-indexed store cannot be promoted to a register or elided, so
     * the array is backed by scratch and flat_scratch is set up.  pad[] is not
     * read after the asm below (see file header). */
    volatile int pad[8];
    int sel = idx[i] & 7;
    pad[sel] = i;

    int   val   = (int)in[i];   /* sign-extended payload to store */
    int   voff  = 16;           /* per-lane scratch byte offset, inside the
                                 * compiler-reserved private region */
    int   result;

    __asm__ volatile(
        "scratch_store_short %1, %2, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "scratch_load_sshort %0, %1, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(result)
        : "v"(voff), "v"(val)
        : "memory");

    out[i] = result;   /* self-contained round-trip; pad[] intentionally unread */
}

int
main(void)
{
    int16_t *h_in  = (int16_t *)malloc(N * sizeof(int16_t));
    int     *h_idx = (int     *)malloc(N * sizeof(int));
    int     *h_out = (int     *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        /* Mix of positive and negative shorts to exercise sign extension. */
        h_in[i]  = (int16_t)((i * 277 + 13) - 32768);
        h_idx[i] = i;
    }

    int16_t *d_in;
    int     *d_idx, *d_out;
    GPU_CHECK(hipMalloc(&d_in,  N * sizeof(int16_t)));
    GPU_CHECK(hipMalloc(&d_idx, N * sizeof(int)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_in,  h_in,  N * sizeof(int16_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_idx, h_idx, N * sizeof(int),     hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_scratch_load_sshort, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_in, d_idx, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int expected = (int)h_in[i];   /* sign-extended 16->32 */
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
