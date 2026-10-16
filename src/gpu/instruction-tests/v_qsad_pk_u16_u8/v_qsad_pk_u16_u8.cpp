/*
 * HIP test for V_QSAD_PK_U16_U8 (VOP3, packed quad sum-of-absolute-diffs).
 *
 * V_QSAD_PK_U16_U8 computes four 4-byte SADs over sliding byte windows of a
 * 64-bit reference operand, each accumulated with a 16-bit lane of a packed
 * accumulator, and packs the four u16 results into a 64-bit destination:
 *
 *   tmp[15:0]  = SAD_U8(S0[31:0],  S1[31:0], S2[15:0])
 *   tmp[31:16] = SAD_U8(S0[39:8],  S1[31:0], S2[31:16])
 *   tmp[47:32] = SAD_U8(S0[47:16], S1[31:0], S2[47:32])
 *   tmp[63:48] = SAD_U8(S0[55:24], S1[31:0], S2[63:48])
 *   D0.b64 = tmp.b64
 *
 * where SAD_U8(a, b, acc) = acc + sum over 4 byte lanes of |a.byte - b.byte|.
 *
 * src0 = 64-bit (8 bytes) reference, src1 = 32-bit (4 bytes) compared window,
 * src2 = 64-bit packed accumulator of four u16.  The four windows slide one
 * byte at a time through the 8-byte reference (bytes [0..3], [1..4], [2..5],
 * [3..6]).
 *
 * The compiler will not naturally emit this instruction, so inline asm is
 * used.  The 64-bit src0/src2 use the "v" constraint on uint64_t (VGPR pair);
 * the 32-bit src1 uses "v" on uint32_t; the 64-bit dst uses "=v" on uint64_t.
 *
 * Inputs are chosen so that no u16 lane overflows: accumulators are small and
 * per-byte absolute differences are bounded, keeping each packed result well
 * under 65535.
 *
 * Build:
 *   make -C hip_tests/v_qsad_pk_u16_u8 AMDGPU_TARGET=gfx950
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_qsad_pk_u16_u8
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,   \
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N          = 256;
static const int BLOCK_SIZE = 64;

/*
 * kernel_qsad
 *
 * Per-thread: load a 64-bit reference (src0), a 32-bit window (src1) and a
 * 64-bit packed accumulator (src2), execute v_qsad_pk_u16_u8, and store the
 * 64-bit packed result.  All operands are VGPR-resident ("v" / "=v").
 */
__global__ void
kernel_qsad(const uint64_t *ref, const uint32_t *win, const uint64_t *acc,
            uint64_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t s0 = ref[i];
    uint32_t s1 = win[i];
    uint64_t s2 = acc[i];

    uint64_t vd;
    __asm__ volatile(
        "v_qsad_pk_u16_u8 %0, %1, %2, %3\n\t"
        : "=v"(vd)
        : "v"(s0), "v"(s1), "v"(s2));

    out[i] = vd;
}

/* Host reference matching the gem5 / AMD ISA definition. */
static uint32_t
sad_u8(uint32_t a, uint32_t b, uint32_t acc)
{
    uint32_t tmp = acc;
    for (int byte = 0; byte < 4; ++byte) {
        uint32_t ab = (a >> (8 * byte)) & 0xff;
        uint32_t bb = (b >> (8 * byte)) & 0xff;
        tmp += (ab > bb) ? (ab - bb) : (bb - ab);
    }
    return tmp;
}

static uint64_t
qsad_ref(uint64_t s0, uint32_t s1, uint64_t s2)
{
    /* Four sliding 4-byte windows of the 8-byte reference: bytes starting at
     * offsets 0, 1, 2, 3 (i.e. bit offsets 0, 8, 16, 24). */
    uint32_t w0 = (uint32_t)(s0 >> 0);
    uint32_t w1 = (uint32_t)(s0 >> 8);
    uint32_t w2 = (uint32_t)(s0 >> 16);
    uint32_t w3 = (uint32_t)(s0 >> 24);

    uint32_t a0 = (uint32_t)((s2 >> 0) & 0xffff);
    uint32_t a1 = (uint32_t)((s2 >> 16) & 0xffff);
    uint32_t a2 = (uint32_t)((s2 >> 32) & 0xffff);
    uint32_t a3 = (uint32_t)((s2 >> 48) & 0xffff);

    uint64_t r0 = sad_u8(w0, s1, a0) & 0xffff;
    uint64_t r1 = sad_u8(w1, s1, a1) & 0xffff;
    uint64_t r2 = sad_u8(w2, s1, a2) & 0xffff;
    uint64_t r3 = sad_u8(w3, s1, a3) & 0xffff;

    return (r3 << 48) | (r2 << 32) | (r1 << 16) | r0;
}

int
main(void)
{
    uint64_t *h_ref = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint32_t *h_win = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint64_t *h_acc = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_out = (uint64_t *)malloc(N * sizeof(uint64_t));

    /* Initialize with varied byte data.  Absolute per-byte diffs are <=255,
     * so each SAD adds at most 4*255=1020; accumulators kept small so no u16
     * lane overflows. */
    for (int i = 0; i < N; i++) {
        uint64_t r = 0;
        for (int byte = 0; byte < 8; ++byte) {
            uint64_t v = (uint64_t)((i * 7 + byte * 31 + 3) & 0xff);
            r |= (v << (8 * byte));
        }
        h_ref[i] = r;

        uint32_t w = 0;
        for (int byte = 0; byte < 4; ++byte) {
            uint32_t v = (uint32_t)((i * 5 + byte * 17 + 11) & 0xff);
            w |= (v << (8 * byte));
        }
        h_win[i] = w;

        uint64_t a = 0;
        for (int lane = 0; lane < 4; ++lane) {
            uint64_t v = (uint64_t)((i + lane * 3) & 0x3f); /* small: <=63 */
            a |= (v << (16 * lane));
        }
        h_acc[i] = a;
    }

    uint64_t *d_ref, *d_acc, *d_out;
    uint32_t *d_win;
    GPU_CHECK(hipMalloc(&d_ref, N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_win, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_acc, N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint64_t)));

    GPU_CHECK(hipMemcpy(d_ref, h_ref, N * sizeof(uint64_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_win, h_win, N * sizeof(uint32_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_acc, h_acc, N * sizeof(uint64_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_qsad, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_ref, d_win, d_acc, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint64_t expected = qsad_ref(h_ref[i], h_win[i], h_acc[i]);
        if (h_out[i] != expected) {
            fprintf(stderr,
                    "FAIL out[%d]: got 0x%016llx expected 0x%016llx\n", i,
                    (unsigned long long)h_out[i],
                    (unsigned long long)expected);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=0x%016llx out[%d]=0x%016llx\n",
               (unsigned long long)h_out[0], N - 1,
               (unsigned long long)h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_ref));
    GPU_CHECK(hipFree(d_win));
    GPU_CHECK(hipFree(d_acc));
    GPU_CHECK(hipFree(d_out));
    free(h_ref);
    free(h_win);
    free(h_acc);
    free(h_out);

    return pass ? 0 : 1;
}
