/*
 * HIP test for BUFFER_STORE_FORMAT_XYZW (ENC_MUBUF, typed 4-component buffer
 * store).
 *
 * With a 32_32_32_32 UINT buffer format and identity dst_sel, this stores four
 * 32-bit components {x,y,z,w} from 4 consecutive VGPRs to a buffer resource,
 * i.e. an identity copy of 16 bytes per record.  A per-thread byte offset
 * (offen) selects each 16-byte record; the stored bytes are read back with a
 * plain host copy and compared.
 *
 * gem5 status note: the MUBUF BUFFER_STORE_FORMAT_* family has no execute()
 * implementation in gem5 (only the MTBUF/TBUFFER variants do), so this test is
 * expected to abort the simulator with "unimplemented VEGA instruction".  It is
 * provided as a ready self-checking artifact once gem5 implements it; the
 * verification is exact for a correct identity-format implementation.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o buffer_store_format_xyzw buffer_store_format_xyzw.cpp
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
kernel_buffer_store_format_xyzw(const uint32_t *src, uint32_t *buf, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    u32x4 data;
    data.x = src[i * 4 + 0];
    data.y = src[i * 4 + 1];
    data.z = src[i * 4 + 2];
    data.w = src[i * 4 + 3];

    uint64_t base = (uint64_t)buf;
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);
    srd.y = (uint32_t)((base >> 32) & 0xffffu);
    srd.z = 0xffffffffu;
    /* word3: dstSel identity (X=4,Y=5,Z=6,W=7), numFmt=UINT(4),
     * dataFmt=32_32_32_32(14), type=0. */
    srd.w = (4u) | (5u << 3) | (6u << 6) | (7u << 9) | (4u << 12) | (14u << 15);

    uint32_t voffset = (uint32_t)(i * 16);

    __asm__ volatile(
        "buffer_store_format_xyzw %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :
        : "v"(data), "v"(voffset), "s"(srd)
        : "memory");
}

int
main(void)
{
    const int total = N * 4;
    uint32_t *h_src = (uint32_t *)malloc(total * sizeof(uint32_t));
    uint32_t *h_buf = (uint32_t *)malloc(total * sizeof(uint32_t));

    for (int i = 0; i < total; i++)
        h_src[i] = (uint32_t)(i * 2654435761u + 12345u);

    uint32_t *d_src, *d_buf;
    GPU_CHECK(hipMalloc(&d_src, total * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_buf, total * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_buf, 0, total * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_src, h_src, total * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_buffer_store_format_xyzw, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_src, d_buf, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_buf, d_buf, total * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < total; i++) {
        if (h_buf[i] != h_src[i]) {
            fprintf(stderr, "FAIL buf[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_buf[i], h_src[i]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("buf[0]=0x%08x buf[%d]=0x%08x\n", h_buf[0], total - 1, h_buf[total - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_buf));
    free(h_src); free(h_buf);
    return pass ? 0 : 1;
}
