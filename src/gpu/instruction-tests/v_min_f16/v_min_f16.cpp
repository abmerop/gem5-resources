/*
 * HIP test for V_MIN_F16 (VOP2 half-precision minimum), with its DPP and
 * SDWA variants.
 *
 * V_MIN_F16 computes:
 *   D.f16 = min(S0.f16, S1.f16)        (IEEE minNum: NaN-suppressing)
 *
 * gem5 implements this in src/arch/amdgpu/common/insts/vop2.cc via
 * fplibMin on the low 16 bits of each source operand, dispatched through
 * vop2Helper which applies the SDWA src/dst selects and DPP lane swizzle
 * to src0 before the operation.  To keep verification exact, all inputs
 * are chosen as exactly-representable f16 values with no NaN/Inf, so the
 * result is always bit-identical to one of the operands.
 *
 * Three kernels exercise three encodings, each self-checked:
 *
 *   base : v_min_f16_e32 vd, va, vb
 *            out = max(a[i], b[i])
 *
 *   dpp  : v_min_f16_dpp vd, va, vb quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
 *            DPP gathers src0 from a swizzled lane within each quad of 4
 *            lanes (lane reads from quadBase + perm[lane&3], perm={1,0,3,2}),
 *            so out = max(a[swizzled_lane], b[i]).
 *
 *   sdwa : v_min_f16_sdwa vd, vpacked, vb dst_sel:WORD_0
 *              dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0
 *            src0_sel:WORD_1 reads the HIGH 16 bits of the packed src0
 *            register (where we stash c[i]), so out = max(c[i], b[i]).
 *            dst_sel:WORD_0 writes the result to the low 16 bits and
 *            UNUSED_PAD zeroes the rest.  (gem5 only allows UNUSED_PRESERVE
 *            with dst_sel:DWORD, so UNUSED_PAD is used for sub-word writes.)
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_min_f16 v_min_f16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_min_f16
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

/* base: v_min_f16_e32 vd, va, vb  -> out = max(a, b) */
__global__ void
kernel_base(const __half *a, const __half *b, __half *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint16_t ha, hb;
    __builtin_memcpy(&ha, a + i, sizeof(ha));
    __builtin_memcpy(&hb, b + i, sizeof(hb));

    uint32_t va = ha, vb = hb, vd;
    __asm__ volatile("v_min_f16_e32 %0, %1, %2\n\t"
                     : "=v"(vd)
                     : "v"(va), "v"(vb));

    uint16_t hd = (uint16_t)(vd & 0xffff);
    __builtin_memcpy(out + i, &hd, sizeof(hd));
}

/*
 * dpp: v_min_f16_dpp vd, va, vb quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
 * src0 (va) is gathered from a swizzled lane within each quad of 4 lanes.
 */
__global__ void
kernel_dpp(const __half *a, const __half *b, __half *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint16_t ha, hb;
    __builtin_memcpy(&ha, a + i, sizeof(ha));
    __builtin_memcpy(&hb, b + i, sizeof(hb));

    uint32_t va = ha, vb = hb, vd;
    __asm__ volatile(
        "v_min_f16_dpp %0, %1, %2 quad_perm:[1,0,3,2] "
        "row_mask:0xf bank_mask:0xf\n\t"
        : "=v"(vd)
        : "v"(va), "v"(vb));

    uint16_t hd = (uint16_t)(vd & 0xffff);
    __builtin_memcpy(out + i, &hd, sizeof(hd));
}

/*
 * sdwa: v_min_f16_sdwa vd, vpacked, vb ... src0_sel:WORD_1 ...
 * c[i] is stashed in the HIGH 16 bits of the packed src0 register and
 * selected via src0_sel:WORD_1, so out = max(c[i], b[i]).
 */
__global__ void
kernel_sdwa(const __half *c, const __half *b, __half *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint16_t hc, hb;
    __builtin_memcpy(&hc, c + i, sizeof(hc));
    __builtin_memcpy(&hb, b + i, sizeof(hb));

    uint32_t packed = ((uint32_t)hc) << 16;   /* hi = c, lo = 0 */
    uint32_t vb = hb;
    uint32_t vd;
    __asm__ volatile(
        "v_min_f16_sdwa %0, %1, %2 dst_sel:WORD_0 "
        "dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0\n\t"
        : "=v"(vd)
        : "v"(packed), "v"(vb));

    uint16_t hd = (uint16_t)(vd & 0xffff);
    __builtin_memcpy(out + i, &hd, sizeof(hd));
}

static uint16_t
half_bits(__half h)
{
    uint16_t b;
    __builtin_memcpy(&b, &h, sizeof(b));
    return b;
}

int
main(void)
{
    __half *h_a   = (__half *)malloc(N * sizeof(__half));
    __half *h_b   = (__half *)malloc(N * sizeof(__half));
    __half *h_c   = (__half *)malloc(N * sizeof(__half));
    __half *h_base = (__half *)malloc(N * sizeof(__half));
    __half *h_dpp  = (__half *)malloc(N * sizeof(__half));
    __half *h_sdwa = (__half *)malloc(N * sizeof(__half));

    /* Exactly-representable f16 values (multiples of 0.25), no NaN/Inf.
     * Ranges overlap so both operands win in different lanes. */
    for (int i = 0; i < N; i++) {
        h_a[i] = __float2half((float)(i % 13) * 0.5f);
        h_b[i] = __float2half((float)(i % 7) * 0.5f - 1.0f);
        h_c[i] = __float2half((float)(i % 5) * 0.25f + 0.5f);
    }

    __half *d_a, *d_b, *d_c, *d_base, *d_dpp, *d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,    N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_b,    N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_c,    N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_base, N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_dpp,  N * sizeof(__half)));
    GPU_CHECK(hipMalloc(&d_sdwa, N * sizeof(__half)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(__half), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(__half), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, N * sizeof(__half), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_base, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_base, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kernel_dpp, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_dpp, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kernel_sdwa, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_c, d_b, d_sdwa, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_base, d_base, N * sizeof(__half),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp, d_dpp, N * sizeof(__half),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa, d_sdwa, N * sizeof(__half),
                        hipMemcpyDeviceToHost));

    /* quad_perm:[1,0,3,2] -> within a quad, lane reads from perm[lane&3]. */
    static const int perm[4] = {1, 0, 3, 2};

    int pass = 1;
    for (int i = 0; i < N; i++) {
        float fa = __half2float(h_a[i]);
        float fb = __half2float(h_b[i]);
        float fc = __half2float(h_c[i]);

        /* base */
        uint16_t exp_base = half_bits(__float2half(fminf(fa, fb)));
        if (half_bits(h_base[i]) != exp_base) {
            fprintf(stderr, "FAIL base[%d]: got 0x%04x exp 0x%04x\n",
                    i, half_bits(h_base[i]), exp_base);
            pass = 0; break;
        }

        /* dpp: src0 from swizzled lane within the same wavefront */
        int lane     = i % BLOCK_SIZE;
        int wavebase = i - lane;
        int srclane  = (lane & ~3) + perm[lane & 3];
        float fa_sw  = __half2float(h_a[wavebase + srclane]);
        uint16_t exp_dpp = half_bits(__float2half(fminf(fa_sw, fb)));
        if (half_bits(h_dpp[i]) != exp_dpp) {
            fprintf(stderr, "FAIL dpp[%d]: got 0x%04x exp 0x%04x\n",
                    i, half_bits(h_dpp[i]), exp_dpp);
            pass = 0; break;
        }

        /* sdwa: src0 high half holds c[i] */
        uint16_t exp_sdwa = half_bits(__float2half(fminf(fc, fb)));
        if (half_bits(h_sdwa[i]) != exp_sdwa) {
            fprintf(stderr, "FAIL sdwa[%d]: got 0x%04x exp 0x%04x\n",
                    i, half_bits(h_sdwa[i]), exp_sdwa);
            pass = 0; break;
        }
    }

    if (pass) {
        printf("base[1]=%.3f dpp[1]=%.3f sdwa[1]=%.3f\n",
               __half2float(h_base[1]), __half2float(h_dpp[1]),
               __half2float(h_sdwa[1]));
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a));    GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c));    GPU_CHECK(hipFree(d_base));
    GPU_CHECK(hipFree(d_dpp));  GPU_CHECK(hipFree(d_sdwa));
    free(h_a); free(h_b); free(h_c);
    free(h_base); free(h_dpp); free(h_sdwa);

    return pass ? 0 : 1;
}
