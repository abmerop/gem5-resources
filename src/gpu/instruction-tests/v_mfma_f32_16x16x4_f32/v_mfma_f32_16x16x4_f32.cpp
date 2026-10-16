/*
 * HIP test for v_mfma_f32_16x16x4_f32 (VOP3P-MAI, CDNA4 / gfx950).
 *
 * This is a matrix fused-multiply-add that computes a single (B=1) 16x16
 * output block with an inner (K) dimension of 4, using single-precision f32
 * A/B inputs and an f32 accumulator:
 *
 *   D[i][j] = C[i][j] + sum_{k=0}^{K-1} A[i][k] * B[k][j]     (i,j in [0,15])
 *
 * In gem5 this maps to:
 *   Inst_VOP3P_MAI__V_MFMA<_delta=1, M=16, N=16, K=4, B=1,
 *                          ConstVecOperandF32, VecOperandF32, ...>
 * (see src/arch/amdgpu/vega/insts/vop3p_mai.hh).
 *
 * Register / lane layout (derived from the gem5 template with these params)
 * ------------------------------------------------------------------------
 *   gprs_a   = M*K*B/64 = 1   -> SRC0 "A" is a single VGPR (one f32/lane)
 *   gprs_b   = K*N*B/64 = 1   -> SRC1 "B" is a single VGPR (one f32/lane)
 *   gprs_c_d = M*N*B/64 = 4   -> SRC2/VDST "C/D" is 4 f32 regs/lane
 *
 *   H   = 4                          (F32 path, _delta==1)
 *   B_I = ceil(64/(N*M/H)) = 1       (= ceil(64/64))
 *   M_I = (64/B_I)/N        = 4
 *   G   = M/(H*M_I)         = 1
 *   K_L = K/(64/(M*B))      = 4/(64/16) = 1
 *
 * Because K_L=1, the K reduction is spread ACROSS wavefront lanes rather than
 * across items within a VGPR: for a given k, the operand comes from a
 * different lane.  With block=0, B=1:
 *
 *   item(input) = k % K_L      = 0            (single f32 per src VGPR)
 *   lane_A      = i + M*(k)    = i + 16*k     (i in [0,15], k in [0,3])
 *   lane_B      = j + N*(k)    = j + 16*k
 *   result[i][j] += srcA[lane_A] * srcB[lane_B]
 *
 * OUTPUT mapping (which of the 4 VGPR "items" and which "lane" holds D[i][j]):
 *   item = (i % H) + H*( i/(H*M_I) + G*(block/B_I) )
 *        = (i % 4) + 4*( i/16 + 0 )
 *        = i % 4                       (i in [0,15] -> item in {0,1,2,3})
 *   lane = j + N*( (i/H) % M_I + M_I*(block % B_I) )
 *        = j + 16*( (i/4) % 4 )
 *        = j + 16*(i/4)                (i/4 in {0,1,2,3}, j in [0,15])
 *
 * The host reference below reproduces this mapping bit-for-bit, so a passing
 * result means the gem5 execute() semantics agree with hardware for this test.
 *
 * The accumulator is placed in AGPRs ("+a") exactly like the proven
 * v_mfma_f32_32x32x1_2b_f32 test.  Small integer-valued f32 inputs keep the
 * matmul exact (no rounding), so comparison is exact (strict ==).
 *
 * Build (CDNA4 only):
 *   make AMDGPU_TARGET=gfx950
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_mfma_f32_16x16x4_f32
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* M=16, N=16, K=4, B=1 */
static const int M = 16;
static const int N = 16;
static const int K = 4;
static const int B = 1;

/* 4 f32 accumulator regs per lane. */
static const int CD_REGS = M * N * B / 64;  /* = 4 */

typedef __attribute__((ext_vector_type(4))) float float4_vec;

/*
 * Kernel: one block = one wavefront (64 threads/lanes).
 *
 *   srcA[base]         : this lane's single f32 A operand (SRC0, 1 VGPR)
 *   srcB[base]         : this lane's single f32 B operand (SRC1, 1 VGPR)
 *   c_init[base*4 + i] : 4 f32 C accumulator values for this lane (SRC2)
 *   d_out[base*4 + i]  : 4 f32 D results for this lane (VDST)
 */
__global__ void
mfma_kernel(const float *srcA, const float *srcB, const float *c_init,
            float *d_out, int num_waves)
{
    int wf_id = blockIdx.x;
    int lane = threadIdx.x;
    if (wf_id >= num_waves)
        return;

    int base = wf_id * 64 + lane;

    float va = srcA[base];
    float vb = srcB[base];

    /* Load 4 initial C values into the AGPR-backed accumulator vector. */
    float4_vec vd;
    for (int i = 0; i < CD_REGS; i++)
        vd[i] = c_init[base * CD_REGS + i];

    /*
     * v_mfma_f32_16x16x4_f32 a[0:3], vA, vB, a[0:3]
     *   SRC0 = va  (single VGPR, f32)
     *   SRC1 = vb  (single VGPR, f32)
     *   SRC2/VDST = vd (4 AGPRs, f32)
     *
     * "+a" marks vd as a read-write AGPR operand; the compiler allocates
     * a[0:3] and initialises them from vd before the MFMA.  "v" places
     * va/vb in single VGPRs.  s_waitcnt vmcnt(0) ensures the prior global
     * loads of va/vb complete before the MFMA reads them.
     */
    /*
     * The trailing s_nop's satisfy the CDNA4 MFMA write->read hazard.
     * Per the CDNA4 ISA "Dependency Resolution" table (section 7.6,
     * Table 38), an SGEMM (F32 MFMA) write to a vDst that is then read by a
     * VM/LDS/FLAT/Export op which *overlaps* the vDst has "no internal
     * forwarding path" and requires manually inserted wait states before the
     * consumer may read the committed result.  v_mfma_f32_16x16x4_f32 is a
     * 32-cycle / 8-pass op, which needs ~10 wait states.
     *
     * The compiler does NOT insert these when the MFMA is written in inline
     * asm and its accumulator is consumed directly by the following
     * global_store (global_store_dwordx4 %0, a[0:3]).  Without the wait, the
     * store samples the later accumulator registers (a[2]/a[3]) before the
     * MFMA has committed them, so it reads the stale pre-MFMA C values -
     * observed on hardware as reg2/reg3 returning C instead of D while the
     * earlier-committed reg0/reg1 are correct.  Two s_nop 15's give 32 wait
     * states, comfortably covering the requirement.
     */
    __asm__ volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "v_mfma_f32_16x16x4_f32 %0, %1, %2, %0\n\t"
        "s_nop 15\n\t"
        "s_nop 15\n\t"
        : "+a"(vd)
        : "v"(va), "v"(vb));

    for (int i = 0; i < CD_REGS; i++)
        d_out[base * CD_REGS + i] = vd[i];
}

/* --------------------------------------------------------------------------
 * Host reference that reproduces the gem5 template layout EXACTLY.
 *
 * srcA_wave / srcB_wave : 64-lane arrays (one f32 per lane)
 * cin_wave              : 64*4 array (row = lane, col = accum reg index)
 * dref_wave            : 64*4 array to be filled with expected D
 * -------------------------------------------------------------------------- */
static void
mfma_reference(const float *srcA_wave, const float *srcB_wave,
               const float *cin_wave, float *dref_wave)
{
    const int H = 4;
    const int B_I = 1;
    const int M_I = 4;
    const int G = 1;
    const int K_L = 1;
    const int block = 0;

    /* Start from C (identity copy), then overwrite the touched positions. */
    for (int t = 0; t < 64 * CD_REGS; t++)
        dref_wave[t] = cin_wave[t];

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            /* Output location of D[i][j] (== C location, accumulate). */
            int item = (i % H) + H * (i / (H * M_I) + G * (block / B_I));
            int lane = j + N * ((i / H) % M_I + M_I * (block % B_I));

            float acc = cin_wave[lane * CD_REGS + item];

            /* K reduction is spread across lanes (K_L=1 -> item(in)=0). */
            for (int k = 0; k < K; ++k) {
                int lane_A = i + M * (block + B * (k / K_L));
                int lane_B = j + N * (block + B * (k / K_L));
                acc += srcA_wave[lane_A] * srcB_wave[lane_B];
            }

            dref_wave[lane * CD_REGS + item] = acc;
        }
    }
}

int
main(void)
{
    const int num_waves = 4;
    const int wf_size = 64;
    const int threads = num_waves * wf_size;

    size_t ab_elems = threads;                 /* 1 f32 per lane */
    size_t cd_elems = (size_t)threads * CD_REGS;

    float *h_a = (float *)calloc(ab_elems, sizeof(float));
    float *h_b = (float *)calloc(ab_elems, sizeof(float));
    float *h_c = (float *)calloc(cd_elems, sizeof(float));
    float *h_d = (float *)calloc(cd_elems, sizeof(float));
    float *h_ref = (float *)calloc(cd_elems, sizeof(float));

    /*
     * Deterministic small integer-valued f32 inputs so the product/sum is
     * exact.  Per lane l (0..63):
     *   A[l] = (l % 7) - 3      -> {-3..3}
     *   B[l] = ((l * 2) % 5) - 2
     * C[base*4 + i] = ((base + i) % 11) - 5
     * All values are small integers; f32 arithmetic is exact.
     */
    for (int t = 0; t < threads; t++) {
        h_a[t] = (float)((t % 7) - 3);
        h_b[t] = (float)(((t * 2) % 5) - 2);
        for (int i = 0; i < CD_REGS; i++)
            h_c[t * CD_REGS + i] = (float)(((t + i) % 11) - 5);
    }

    float *d_a, *d_b, *d_c, *d_d;
    GPU_CHECK(hipMalloc(&d_a, ab_elems * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_b, ab_elems * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_c, cd_elems * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_d, cd_elems * sizeof(float)));

    GPU_CHECK(hipMemcpy(d_a, h_a, ab_elems * sizeof(float),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, ab_elems * sizeof(float),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c, h_c, cd_elems * sizeof(float),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d, 0, cd_elems * sizeof(float)));

    hipLaunchKernelGGL(mfma_kernel, dim3(num_waves), dim3(wf_size), 0, 0,
                       d_a, d_b, d_c, d_d, num_waves);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_d, d_d, cd_elems * sizeof(float),
                        hipMemcpyDeviceToHost));

    /* Build the reference wave-by-wave. */
    for (int wf = 0; wf < num_waves; wf++) {
        int off = wf * wf_size;
        mfma_reference(&h_a[off], &h_b[off], &h_c[off * CD_REGS],
                       &h_ref[off * CD_REGS]);
    }

    int pass = 1;
    int checked = 0;
    for (size_t t = 0; t < cd_elems && pass; t++) {
        float got = h_d[t];
        float exp = h_ref[t];
        /* Integer-valued exact math: require exact equality. */
        if (got != exp) {
            int lane = (int)(t / CD_REGS) % wf_size;
            int wf = (int)(t / CD_REGS) / wf_size;
            int reg = (int)(t % CD_REGS);
            fprintf(stderr,
                    "FAIL wf=%d lane=%d reg=%d: got %.6f expected %.6f\n",
                    wf, lane, reg, got, exp);
            pass = 0;
            break;
        }
        checked++;
    }

    printf("v_mfma_f32_16x16x4_f32 test\n");
    printf("  M=%d N=%d K=%d B=%d  (CDNA4/gfx950)\n", M, N, K, B);
    printf("  Checked %d accumulator values\n", checked);
    if (pass) {
        printf("  D[0]=%.1f  D[last]=%.1f\n", h_d[0], h_d[cd_elems - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a));
    GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c));
    GPU_CHECK(hipFree(d_d));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_d);
    free(h_ref);

    return pass ? 0 : 1;
}
