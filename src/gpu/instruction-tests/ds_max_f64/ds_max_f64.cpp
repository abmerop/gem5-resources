/*
 * HIP test for DS_MAX_F64 (DS / LDS atomic, double-precision max, non-RTN).
 *
 * DS_MAX_F64 computes (per the gem5 .arch description):
 *   tmp = MEM[ADDR];        // 64b LDS location
 *   src = DATA;             // f64 data operand
 *   MEM[ADDR] = (tmp > src) ? tmp : src;   // f64 maximum, NaN/INF/denorm aware
 * This is the non-return form: no RETURN_DATA is produced.
 *
 * Test strategy:
 *   Each block (one wavefront, 64 lanes) has a __shared__ double initialized
 *   by lane 0 to a known base value. Every lane then atomically max-es its own
 *   f64 input value into that single LDS location via ds_max_f64 (inline asm,
 *   non-RTN). After a __syncthreads() (which lowers to s_barrier + the required
 *   s_waitcnt lgkmcnt(0) so the LDS write is visible) lane 0 reads the LDS slot
 *   back and stores it to global output, one result per block.
 *
 *   The instruction is emitted via inline asm on an LDS address operand
 *   ("v"(uint32_t lds_offset)) because the compiler will not otherwise pick the
 *   64-bit float LDS atomic max.
 *
 * Host reference: for each block, max(base, all lane inputs).
 *
 * Build:
 *   make -C hip_tests/ds_max_f64 AMDGPU_TARGET=gfx950
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./ds_max_f64
 */

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                     \
    do {                                                                   \
        hipError_t e = (cmd);                                             \
        if (e != hipSuccess) {                                            \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__, \
                    hipGetErrorString(e));                               \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

static const int BLOCK_SIZE = 64;   /* one wavefront per block */
static const int NUM_BLOCKS = 4;
static const int N          = NUM_BLOCKS * BLOCK_SIZE;

/*
 * kernel_ds_max_f64
 *
 * One __shared__ double per block. Lane 0 seeds it with `base`, then every
 * lane atomically maxes its input into it using ds_max_f64 (non-RTN). The
 * LDS address operand is the byte offset of the shared variable (a VGPR-held
 * uint32_t). After the barrier, lane 0 writes the LDS value to global memory.
 */
__global__ void
kernel_ds_max_f64(const double *in, double *out, double base)
{
    __shared__ double smax;

    int lane = threadIdx.x;
    int blk  = blockIdx.x;

    if (lane == 0)
        smax = base;
    __syncthreads();

    double val = in[blk * BLOCK_SIZE + lane];

    /* Byte offset of smax within LDS -> address VGPR for the DS instruction. */
    uint32_t lds_off = (uint32_t)(uintptr_t)(&smax);

    __asm__ volatile(
        "ds_max_f64 %0, %1\n\t"
        :
        : "v"(lds_off), "v"(val)
        : "memory");

    /* __syncthreads() lowers to s_waitcnt lgkmcnt(0) + s_barrier, making the
     * LDS atomic result visible before lane 0 reads it back. */
    __syncthreads();

    if (lane == 0)
        out[blk] = smax;
}

int
main(void)
{
    const double base = 1.5;

    double *h_in  = (double *)malloc(N * sizeof(double));
    double *h_out = (double *)malloc(NUM_BLOCKS * sizeof(double));

    /* Interesting values: negatives, zeros, small, large, and a per-block
     * unique maximum so the reference is nontrivial. */
    for (int b = 0; b < NUM_BLOCKS; b++) {
        for (int l = 0; l < BLOCK_SIZE; l++) {
            int i = b * BLOCK_SIZE + l;
            /* spread of values; lane l==b*7 gets a big block-specific value */
            double v = (double)(l - 32) * 0.25;
            if (l == (b * 7) % BLOCK_SIZE)
                v = 100.0 + (double)b;   /* block-unique large value */
            h_in[i] = v;
        }
    }

    double *d_in, *d_out;
    GPU_CHECK(hipMalloc(&d_in,  N * sizeof(double)));
    GPU_CHECK(hipMalloc(&d_out, NUM_BLOCKS * sizeof(double)));

    GPU_CHECK(hipMemcpy(d_in, h_in, N * sizeof(double),
                        hipMemcpyHostToDevice));

    hipLaunchKernelGGL(kernel_ds_max_f64,
                       dim3(NUM_BLOCKS), dim3(BLOCK_SIZE), 0, 0,
                       d_in, d_out, base);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, NUM_BLOCKS * sizeof(double),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        double expected = base;
        for (int l = 0; l < BLOCK_SIZE; l++) {
            double v = h_in[b * BLOCK_SIZE + l];
            if (v > expected)
                expected = v;
        }
        if (h_out[b] != expected) {
            fprintf(stderr, "FAIL block %d: got %f expected %f\n",
                    b, h_out[b], expected);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=%f out[%d]=%f\n", h_out[0], NUM_BLOCKS - 1,
               h_out[NUM_BLOCKS - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_in));
    GPU_CHECK(hipFree(d_out));
    free(h_in);
    free(h_out);

    return pass ? 0 : 1;
}
