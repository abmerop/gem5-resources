/*
 * HIP test for V_CVT_SR_F16_F32 (VOP3, stochastic-rounding f32->f16) on
 * CDNA4 (gfx950).
 *
 * Semantics (gem5 src/arch/amdgpu/common/insts/vop3.cc):
 *   VDST[15:0]  = f32_to_f16(SRC0.f32, roundStochastic, seed=SRC1.u32)   (base)
 *   VDST[31:16] = f32_to_f16(SRC0.f32, roundStochastic, seed=SRC1.u32)   (op_sel hi)
 * The unwritten half of VDST is preserved (the instruction read-modifies VDST).
 *
 * Making a stochastic-rounding op self-checking:
 *   Stochastic rounding only perturbs the result when the source value has
 *   non-zero *discarded* mantissa bits.  If SRC0 is EXACTLY representable in
 *   f16, there are no discarded bits, so the result is deterministic and equal
 *   to the plain round-to-nearest conversion, independent of the seed.  We feed
 *   only exactly-representable f32 values (small integers, halves, quarters)
 *   and vary the per-lane seed to prove seed-independence for exact inputs.
 *
 * Two kernels exercise the single "base" mnemonic (both disassemble to
 * v_cvt_sr_f16_f32); op_sel:[0,0,1] just steers the result into the high half.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_cvt_sr_f16_f32 v_cvt_sr_f16_f32.cpp
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N          = 256;
static const int BLOCK_SIZE = 64;

/* f32 values exactly representable in f16 (repeated to fill N). */
static const float EXACT[] = {
    1.0f, 2.0f, 0.5f, -1.0f, 3.0f, 0.25f, -4.0f, 1.5f,
    8.0f, -0.5f, 16.0f, 0.75f, -2.5f, 64.0f, 0.125f, -1.5f,
};
static const int NEXACT = sizeof(EXACT) / sizeof(EXACT[0]);

/* Host reference: exact f32->f16 raw bits via the native _Float16 type. */
static uint16_t
f16_bits(float v)
{
    _Float16 h = (_Float16)v;
    uint16_t b;
    memcpy(&b, &h, sizeof(b));
    return b;
}

/* base: v_cvt_sr_f16_f32 -> low half; high half must be preserved. */
__global__ void
kernel_base(const float *in, const uint32_t *seed, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float s0    = in[i];
    uint32_t s1 = seed[i];
    uint32_t d  = 0xDEAD0000u; /* sentinel; low half will be overwritten */
    __asm__ volatile("v_cvt_sr_f16_f32 %0, %1, %2\n\t"
                     : "+v"(d)
                     : "v"(s0), "v"(s1));
    out[i] = d;
}

/* op_sel hi: v_cvt_sr_f16_f32 ... op_sel:[0,0,1] -> high half; low preserved. */
__global__ void
kernel_hi(const float *in, const uint32_t *seed, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float s0    = in[i];
    uint32_t s1 = seed[i];
    uint32_t d  = 0x0000BEEFu; /* sentinel; high half will be overwritten */
    __asm__ volatile("v_cvt_sr_f16_f32 %0, %1, %2 op_sel:[0,0,1]\n\t"
                     : "+v"(d)
                     : "v"(s0), "v"(s1));
    out[i] = d;
}

int
main(void)
{
    float    *h_in   = (float *)malloc(N * sizeof(float));
    uint32_t *h_seed = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_base = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_hi   = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_in[i]   = EXACT[i % NEXACT];
        h_seed[i] = (uint32_t)(i * 2654435761u) ^ 0x9E3779B9u; /* varied */
    }

    float    *d_in;
    uint32_t *d_seed, *d_base, *d_hi;
    GPU_CHECK(hipMalloc(&d_in,   N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_seed, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_base, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_hi,   N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_in,   h_in,   N * sizeof(float),    hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_seed, h_seed, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_base, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_in, d_seed, d_base, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kernel_hi, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_in, d_seed, d_hi, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_base, d_base, N * sizeof(uint32_t), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_hi,   d_hi,   N * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        uint16_t ref = f16_bits(h_in[i]);

        /* base: low half = ref, high half preserved (0xDEAD). */
        uint16_t got_lo = (uint16_t)(h_base[i] & 0xFFFF);
        uint16_t got_hi = (uint16_t)(h_base[i] >> 16);
        if (got_lo != ref || got_hi != 0xDEAD) {
            fprintf(stderr,
                    "FAIL base[%d] in=%g got=0x%08x expected lo=0x%04x hi=0xDEAD\n",
                    i, h_in[i], h_base[i], ref);
            pass = 0;
            break;
        }

        /* op_sel hi: high half = ref, low half preserved (0xBEEF). */
        uint16_t hi_lo = (uint16_t)(h_hi[i] & 0xFFFF);
        uint16_t hi_hi = (uint16_t)(h_hi[i] >> 16);
        if (hi_hi != ref || hi_lo != 0xBEEF) {
            fprintf(stderr,
                    "FAIL hi[%d] in=%g got=0x%08x expected hi=0x%04x lo=0xBEEF\n",
                    i, h_in[i], h_hi[i], ref);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("base[0]=0x%08x hi[0]=0x%08x base[%d]=0x%08x\n",
               h_base[0], h_hi[0], N - 1, h_base[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_in));
    GPU_CHECK(hipFree(d_seed));
    GPU_CHECK(hipFree(d_base));
    GPU_CHECK(hipFree(d_hi));
    free(h_in);
    free(h_seed);
    free(h_base);
    free(h_hi);

    return pass ? 0 : 1;
}
