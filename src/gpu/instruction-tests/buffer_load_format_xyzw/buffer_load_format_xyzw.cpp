/*
 * HIP test for BUFFER_LOAD_FORMAT_XYZW (ENC_MUBUF, typed 4-component buffer
 * load).
 *
 * With a 32_32_32_32 UINT buffer format and identity dst_sel, this loads four
 * 32-bit components {x,y,z,w} from a buffer resource into 4 consecutive VGPRs,
 * i.e. an identity copy of 16 bytes per record.  The buffer resource descriptor
 * (V#) is built over the source allocation with the format fields set; a
 * per-thread byte offset (offen) selects each 16-byte record.
 *
 * gem5 status note: as of this writing
 * Inst_MUBUF__BUFFER_LOAD_FORMAT_XYZW::execute() is panicUnimplemented(), so
 * this test is expected to abort the simulator.  It is provided as a ready
 * self-checking artifact once gem5 implements the instruction; the verification
 * below is exact for a correct identity-format implementation.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o buffer_load_format_xyzw buffer_load_format_xyzw.cpp
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

__global__ void
kernel_buffer_load_format_xyzw(const uint32_t *src, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t base = (uint64_t)src;
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);            /* base lo           */
    srd.y = (uint32_t)((base >> 32) & 0xffffu);        /* base hi, stride=0 */
    srd.z = 0xffffffffu;                               /* num records       */
    /* word3: dstSel identity (X=4,Y=5,Z=6,W=7), numFmt=UINT(4),
     * dataFmt=32_32_32_32(14), type=0. */
    srd.w = (4u) | (5u << 3) | (6u << 6) | (7u << 9) | (4u << 12) | (14u << 15);

    uint32_t voffset = (uint32_t)(i * 16);             /* per-thread record */
    u32x4 data;

    __asm__ volatile(
        "buffer_load_format_xyzw %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(data)
        : "v"(voffset), "s"(srd)
        : "memory");

    out[i * 4 + 0] = data.x;
    out[i * 4 + 1] = data.y;
    out[i * 4 + 2] = data.z;
    out[i * 4 + 3] = data.w;
}

int
main(void)
{
    const int total = N * 4;
    uint32_t *h_src = (uint32_t *)malloc(total * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(total * sizeof(uint32_t));

    for (int i = 0; i < total; i++)
        h_src[i] = (uint32_t)(i * 2654435761u + 12345u);

    uint32_t *d_src, *d_out;
    GPU_CHECK(hipMalloc(&d_src, total * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, total * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_src, h_src, total * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_buffer_load_format_xyzw, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, total * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < total; i++) {
        if (h_out[i] != h_src[i]) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], h_src[i]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], total - 1, h_out[total - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_out));
    free(h_src); free(h_out);
    return pass ? 0 : 1;
}
