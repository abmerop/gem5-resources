/*
 * HIP test for S_BITSET1_B64 (ENC_SOP1, set one bit of a 64-bit scalar to 1).
 *
 * gem5 (src/arch/amdgpu/common/insts/sop1.cc):
 *   SDST.u64[SSRC0.u[5:0]] = 1
 * gem5 sets the requested bit via replaceBits() on the destination operand
 * buffer *without* reading the prior destination value, so only the target bit
 * is guaranteed; the remaining bits are left as-is by the model.  Therefore
 * this test verifies exactly the defined semantic: after the instruction, the
 * bit numbered SSRC0 must read as 1.
 *
 * Pattern B (scalar): the bit index is a uniform SGPR ("s"), and the 64-bit
 * result is produced via a read/write scalar operand ("+s").
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int BLOCK_SIZE = 64;

__global__ void
kernel_s_bitset1(uint32_t bit, uint64_t seed, uint64_t *out)
{
    uint64_t sdst = seed;
    __asm__ volatile(
        "s_bitset1_b64 %0, %1\n\t"
        : "+s"(sdst)
        : "s"(bit));
    if (threadIdx.x == 0)
        out[blockIdx.x] = sdst;
}

int
main(void)
{
    uint32_t bits[] = {0u, 1u, 7u, 15u, 31u, 32u, 33u, 47u, 62u, 63u};
    const int NC = sizeof(bits) / sizeof(bits[0]);
    /* Seed with a value that already has the target bit CLEAR so the check is
     * meaningful (the instruction must flip it to 1). */
    uint64_t seed = 0x0ull;

    uint64_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, NC * sizeof(uint64_t)));

    for (int c = 0; c < NC; c++)
        kernel_s_bitset1<<<1, BLOCK_SIZE>>>(bits[c], seed, d_out + c);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    uint64_t h_out[64];
    GPU_CHECK(hipMemcpy(h_out, d_out, NC * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int c = 0; c < NC; c++) {
        if (((h_out[c] >> bits[c]) & 1ull) != 1ull) {
            fprintf(stderr, "FAIL[%d]: bit %u not set, result=0x%016llx\n",
                    c, bits[c], (unsigned long long)h_out[c]);
            pass = 0;
        }
    }

    if (pass)
        printf("bit63 result=0x%016llx\n", (unsigned long long)h_out[NC-1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    return pass ? 0 : 1;
}
