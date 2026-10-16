/*
 * HIP test for BUFFER_LOAD_UBYTE (ENC_MUBUF, untyped 1-byte buffer load,
 * zero-extended).
 *
 * Loads one byte from a buffer resource and zero-extends it into the low 8
 * bits of a 32-bit VGPR (upper 24 bits cleared).  A raw buffer resource
 * descriptor (V#) is built over the source allocation; a per-thread byte
 * offset (offen) selects each byte.  Pattern C, base only.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o buffer_load_ubyte buffer_load_ubyte.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./buffer_load_ubyte
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

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;

/*
 * kernel_buffer_load_ubyte
 *
 * Each thread loads src[i] (one byte) via a MUBUF byte load and stores the
 * zero-extended 32-bit result to out[i].
 */
__global__ void
kernel_buffer_load_ubyte(const uint8_t *src, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t base = (uint64_t)src;
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);          /* base lo            */
    srd.y = (uint32_t)((base >> 32) & 0xffffu);      /* base hi, stride=0  */
    srd.z = 0xffffffffu;                             /* num records (bytes)*/
    srd.w = 0x00020000u;                             /* gfx9 raw buffer    */

    uint32_t voffset = (uint32_t)i;                  /* per-thread 1B slot */
    uint32_t data;
    __asm__ volatile(
        "buffer_load_ubyte %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(data)
        : "v"(voffset), "s"(srd)
        : "memory");

    out[i] = data;
}

int
main(void)
{
    uint8_t  *h_src = (uint8_t *)malloc(N * sizeof(uint8_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++)
        h_src[i] = (uint8_t)((i * 37 + 11) & 0xff);

    uint8_t  *d_src;
    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(uint8_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_src, h_src, N * sizeof(uint8_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_buffer_load_ubyte,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t exp = (uint32_t)h_src[i];           /* zero-extended */
        if (h_out[i] != exp) {
            fprintf(stderr, "FAIL[%d] got 0x%08x exp 0x%08x\n",
                    i, h_out[i], exp);
            pass = 0;
            break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], N - 1,
               h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src));
    GPU_CHECK(hipFree(d_out));
    free(h_src);
    free(h_out);

    return pass ? 0 : 1;
}
