/*
 * HIP test for BUFFER_STORE_SHORT_D16_HI (MUBUF, gfx950 / CDNA4).
 *
 * BUFFER_STORE_SHORT_D16_HI stores the HIGH 16 bits of the source data
 * VGPR to memory as a 16-bit value:
 *     stored_short = (srcval >> 16) & 0xFFFF
 * (see src/arch/amdgpu/common/insts/mubuf.cc:
 *  Inst_MUBUF__BUFFER_STORE_SHORT_D16_HI::execute uses bits(data,31,16))
 *
 * The compiler will not emit this narrow high-half store for a plain
 * uint16_t write, so inline asm is required.  Each thread loads a known
 * 32-bit value into a VGPR and stores that value's high half to its own
 * per-thread 16-bit slot of an output buffer via a hand-built 128-bit
 * buffer resource descriptor (V#).
 *
 * Build:
 *   make -C hip_tests/buffer_store_short_d16_hi AMDGPU_TARGET=gfx950
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./buffer_store_short_d16_hi
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                     \
    do {                                                                   \
        hipError_t e = (cmd);                                              \
        if (e != hipSuccess) {                                            \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__, \
                    hipGetErrorString(e));                                 \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;

/* Per-thread 32-bit source value; high half is what gets stored. */
static inline __host__ __device__ uint32_t
srcval(int i)
{
    return 0x89AB1234u + (uint32_t)i * 0x00010007u;
}

/*
 * kernel: build a 128-bit V# describing the uint16_t output buffer, then
 * store the high 16 bits of a known 32-bit value to this thread's slot
 * using buffer_store_short_d16_hi.  offen provides the per-thread byte
 * offset (i * 2).  s_waitcnt vmcnt(0) ensures the store is visible.
 */
__global__ void
kernel(uint16_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t base = (uint64_t)out;
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);           /* base_addr[31:0]  */
    srd.y = (uint32_t)((base >> 32) & 0xffffu);       /* base_addr[47:32] */
    srd.z = 0xffffffffu;                              /* num_records      */
    /* stride/dfmt/nfmt/etc bits (mirrors other buffer_store tests) */
    srd.w = (4u) | (5u << 3) | (6u << 6) | (7u << 9) | (4u << 12) |
            (4u << 15);

    uint32_t val  = srcval(i);
    uint32_t voff = (uint32_t)(i * 2);

    __asm__ volatile(
        "buffer_store_short_d16_hi %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :
        : "v"(val), "v"(voff), "s"(srd)
        : "memory");
}

int
main(void)
{
    uint16_t *h_out = (uint16_t *)malloc(N * sizeof(uint16_t));

    uint16_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint16_t)));
    GPU_CHECK(hipMemset(d_out, 0, N * sizeof(uint16_t)));

    kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(
        hipMemcpy(h_out, d_out, N * sizeof(uint16_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint16_t expected = (uint16_t)((srcval(i) >> 16) & 0xffffu);
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%04x expected 0x%04x\n", i,
                    (unsigned)h_out[i], (unsigned)expected);
            pass = 0;
            break;
        }
    }

    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    free(h_out);

    return pass ? 0 : 1;
}
