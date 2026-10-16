/*
 * HIP test for S_BCNT1_I32_B64 (ENC_SOP1, count set bits in a 64-bit value).
 *
 * gem5 (src/arch/amdgpu/common/insts/sop1.cc) computes:
 *   SDST.i32 = popcount(SSRC0.u64)
 *   SCC      = (SDST != 0)
 *
 * Pattern B (scalar): the uint64_t kernel argument is uniform across the
 * wavefront (SGPR pair); "=s"/"s" force the scalar S_BCNT1_I32_B64 encoding.
 * Population count is exact -> bit-exact verification.
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

static const int BLOCK_SIZE = 64;

__global__ void
kernel_s_bcnt1(uint64_t s0, uint32_t *out)
{
    uint32_t sdst;
    __asm__ volatile(
        "s_bcnt1_i32_b64 %0, %1\n\t"
        : "=s"(sdst)
        : "s"(s0));
    if (threadIdx.x == 0)
        out[blockIdx.x] = sdst;
}

int
main(void)
{
    uint64_t cases[] = {
        0x0000000000000000ull, 0xffffffffffffffffull, 0x0000000000000001ull,
        0x8000000000000000ull, 0x0123456789abcdefull, 0xaaaaaaaaaaaaaaaaull,
        0x5555555555555555ull, 0xdeadbeefcafebabeull, 0x00000000ffffffffull,
        0xf0f0f0f0f0f0f0f0ull,
    };
    const int NC = sizeof(cases) / sizeof(cases[0]);

    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, NC * sizeof(uint32_t)));

    for (int c = 0; c < NC; c++)
        kernel_s_bcnt1<<<1, BLOCK_SIZE>>>(cases[c], d_out + c);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    uint32_t h_out[64];
    GPU_CHECK(hipMemcpy(h_out, d_out, NC * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int c = 0; c < NC; c++) {
        uint32_t expected = __builtin_popcountll(cases[c]);
        if (h_out[c] != expected) {
            fprintf(stderr, "FAIL[%d]: popcnt(%016llx) got %u exp %u\n", c,
                    (unsigned long long)cases[c], h_out[c], expected);
            pass = 0;
        }
    }

    if (pass)
        printf("out[4]=%u\n", h_out[4]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
