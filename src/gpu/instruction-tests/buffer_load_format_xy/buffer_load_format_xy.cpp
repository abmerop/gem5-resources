/*
 * HIP test for BUFFER_LOAD_FORMAT_XY (ENC_MUBUF, typed 2-component buffer
 * load).
 *
 * BUFFER_LOAD_FORMAT_XY loads two format-converted components {x,y} from a
 * buffer resource into two consecutive VGPRs.  The conversion applied is
 * determined by the numeric-format / data-format fields of the buffer
 * resource descriptor (V#).  When the buffer's data format is 32_32 and the
 * numeric format is FLOAT, each component is a raw 32-bit IEEE-754 float and
 * no conversion occurs: the "format" load is effectively an untyped 2xf32
 * load and returns the input floats unchanged.
 *
 * This test builds a 128-bit buffer resource descriptor over the source
 * allocation with:
 *   - base address = src pointer, stride = 0
 *   - num_records  = 0xffffffff (do not out-of-range clamp)
 *   - dst_sel      = identity (X=4, Y=5, Z=6, W=7)
 *   - num_format   = FLOAT (7)
 *   - data_format  = 32_32 (12)
 * and uses a per-thread byte offset (offen) to select each 8-byte record of
 * two floats.  Each thread loads {x,y} and stores them; the host verifies the
 * loaded floats equal the input floats exactly.
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   buffer_load_format_xy v[0:1], v2, s[0:3], 0 offen
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o buffer_load_format_xy \
 *       buffer_load_format_xy.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./buffer_load_format_xy
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
typedef float    f32x2 __attribute__((ext_vector_type(2)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;

/*
 * kernel_buffer_load_format_xy
 *
 * Each thread constructs a buffer resource descriptor (V#) over `src`, sets
 * the data format to 32_32 / FLOAT so the format load is an identity 2xf32
 * load, and issues BUFFER_LOAD_FORMAT_XY with a per-thread 8-byte offset
 * (offen).  The two loaded floats are written back to `out`.
 *
 * s_waitcnt vmcnt(0) stalls until the vector memory load completes before the
 * destination VGPRs are consumed.
 */
__global__ void
kernel_buffer_load_format_xy(const float *src, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t base = (uint64_t)src;
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);            /* base lo           */
    srd.y = (uint32_t)((base >> 32) & 0xffffu);        /* base hi, stride=0 */
    srd.z = 0xffffffffu;                               /* num records       */
    /* word3: dstSel identity (X=4,Y=5,Z=6,W=7), numFmt=FLOAT(7),
     * dataFmt=32_32(12), type=0. */
    srd.w = (4u) | (5u << 3) | (6u << 6) | (7u << 9) | (7u << 12) | (12u << 15);

    uint32_t voffset = (uint32_t)(i * 8);              /* per-thread record */
    f32x2 data;

    __asm__ volatile(
        "buffer_load_format_xy %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(data)
        : "v"(voffset), "s"(srd)
        : "memory");

    out[i * 2 + 0] = data.x;
    out[i * 2 + 1] = data.y;
}

int
main(void)
{
    const int total = N * 2;
    float *h_src = (float *)malloc(total * sizeof(float));
    float *h_out = (float *)malloc(total * sizeof(float));

    for (int i = 0; i < total; i++)
        h_src[i] = (float)i * 1.5f - 3.25f;

    float *d_src, *d_out;
    GPU_CHECK(hipMalloc(&d_src, total * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_out, total * sizeof(float)));
    GPU_CHECK(hipMemcpy(d_src, h_src, total * sizeof(float),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_buffer_load_format_xy, dim3(grid),
                       dim3(BLOCK_SIZE), 0, 0, d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, total * sizeof(float),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < total; i++) {
        if (h_out[i] != h_src[i]) {
            fprintf(stderr, "FAIL out[%d]: got %g expected %g\n",
                    i, h_out[i], h_src[i]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=%g out[%d]=%g\n", h_out[0], total - 1,
               h_out[total - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_out));
    free(h_src); free(h_out);
    return pass ? 0 : 1;
}
