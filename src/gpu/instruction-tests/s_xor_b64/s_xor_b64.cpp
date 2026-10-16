/*
 * HIP test for S_XOR_B64 (ENC_SOP2, scalar 64-bit bitwise XOR).
 *
 * gem5 (src/arch/amdgpu/common/insts/sop2.cc) computes:
 *   SDST.u64 = SSRC0.u64 ^ SSRC1.u64
 *   SCC      = (SDST != 0)
 *
 * Pattern B (scalar): uint64_t kernel arguments are uniform across the
 * wavefront and therefore live in SGPR pairs; "=s"/"s" force the scalar
 * S_XOR_B64 encoding.  Pure bitwise op -> bit-exact verification.
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
kernel_s_xor_b64(uint64_t s0, uint64_t s1, uint64_t *out)
{
    uint64_t sdst;
    __asm__ volatile(
        "s_xor_b64 %0, %1, %2\n\t"
        : "=s"(sdst)
        : "s"(s0), "s"(s1));
    if (threadIdx.x == 0)
        out[blockIdx.x] = sdst;
}

int
main(void)
{
    struct { uint64_t s0, s1; } cases[] = {
        {0x0000000000000000ull, 0x0000000000000000ull},
        {0xffffffffffffffffull, 0x0000000000000000ull},
        {0xffffffffffffffffull, 0xffffffffffffffffull},
        {0x0123456789abcdefull, 0xfedcba9876543210ull},
        {0xdeadbeefcafebabeull, 0x1122334455667788ull},
        {0x8000000000000001ull, 0x8000000000000001ull},
        {0xaaaaaaaaaaaaaaaaull, 0x5555555555555555ull},
        {0x00000000ffffffffull, 0xffffffff00000000ull},
    };
    const int NC = sizeof(cases) / sizeof(cases[0]);

    uint64_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, NC * sizeof(uint64_t)));

    for (int c = 0; c < NC; c++)
        kernel_s_xor_b64<<<1, BLOCK_SIZE>>>(cases[c].s0, cases[c].s1,
                                            d_out + c);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    uint64_t h_out[64];
    GPU_CHECK(hipMemcpy(h_out, d_out, NC * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int c = 0; c < NC; c++) {
        uint64_t expected = cases[c].s0 ^ cases[c].s1;
        if (h_out[c] != expected) {
            fprintf(stderr, "FAIL[%d]: got %016llx exp %016llx\n", c,
                    (unsigned long long)h_out[c],
                    (unsigned long long)expected);
            pass = 0;
        }
    }

    if (pass)
        printf("out[3]=%016llx\n", (unsigned long long)h_out[3]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
