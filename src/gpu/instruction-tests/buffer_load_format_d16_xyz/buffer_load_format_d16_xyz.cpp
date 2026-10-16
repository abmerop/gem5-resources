/*
 * HIP test for BUFFER_LOAD_FORMAT_D16_XYZ (ENC_MUBUF, typed 3-component buffer
 * load into packed 16-bit destinations).  Pattern C.
 *
 * Loads three 16-bit components {x,y,z} from a buffer resource into packed
 * VGPRs (v0 = {x,y}, v1 low = z).  The buffer resource descriptor (V#) is
 * built over a 16-bit source allocation with a 16_16_16_16 / UINT format; a
 * per-thread byte offset (offen) selects each 8-byte record and the low three
 * shorts are read.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   buffer_load_format_d16_xyz v[0:1], v2, s[0:3], 0 offen
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o buffer_load_format_d16_xyz \
 *        buffer_load_format_d16_xyz.cpp
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
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kernel(const uint16_t *src, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t base = (uint64_t)src;
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);
    srd.y = (uint32_t)((base >> 32) & 0xffffu);
    srd.z = 0xffffffffu;
    /* dst_sel identity; nfmt=UINT(4), dfmt=16_16_16_16(12). */
    srd.w = (4u) | (5u << 3) | (6u << 6) | (7u << 9) | (12u << 12) | (4u << 15);

    uint32_t voffset = (uint32_t)(i * 8); /* per-thread 4x16-bit record */
    u32x2 data;
    __asm__ volatile(
        "buffer_load_format_d16_xyz %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(data)
        : "v"(voffset), "s"(srd)
        : "memory");
    /* v0 = {x,y}, v1 low = z */
    out[i * 3 + 0] = data.x & 0xffffu;
    out[i * 3 + 1] = (data.x >> 16) & 0xffffu;
    out[i * 3 + 2] = data.y & 0xffffu;
}

int
main(void)
{
    uint16_t *h_src = (uint16_t *)malloc(N * 4 * sizeof(uint16_t));
    uint32_t *h_out = (uint32_t *)malloc(N * 3 * sizeof(uint32_t));
    for (int i = 0; i < N * 4; i++)
        h_src[i] = (uint16_t)(i * 1103 + 7);
    uint16_t *d_src;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_src, N * 4 * sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_out, N * 3 * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_src, h_src, N * 4 * sizeof(uint16_t),
                        hipMemcpyHostToDevice));
    kernel<<<N / BLOCK_SIZE, BLOCK_SIZE>>>(d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out, d_out, N * 3 * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N; i++) {
        for (int c = 0; c < 3; c++) {
            uint32_t exp = (uint32_t)h_src[i * 4 + c];
            if (h_out[i * 3 + c] != exp) {
                fprintf(stderr, "FAIL[%d,%d] got 0x%x exp 0x%x\n", i, c,
                        h_out[i * 3 + c], exp);
                pass = 0;
                break;
            }
        }
        if (!pass)
            break;
    }
    if (pass)
        printf("out[0]=0x%x out[%d]=0x%x\n", h_out[0], N * 3 - 1,
               h_out[N * 3 - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
