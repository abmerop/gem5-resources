/*
 * HIP test for BUFFER_LOAD_FORMAT_D16_XY (ENC_MUBUF, typed 2-component D16
 * buffer load).
 *
 * With a 16_16 UINT buffer format, this loads two 16-bit components {x,y} from a
 * buffer resource and packs them into the low/high halves of a single 32-bit
 * VGPR (D16), i.e. an identity copy of 4 bytes per record.  The buffer resource
 * descriptor (V#) is built over the source allocation with the format fields
 * set; a per-thread byte offset (offen) selects each 4-byte record.
 *
 * gem5 status note: the MUBUF BUFFER_LOAD_FORMAT_* family has no execute()
 * implementation in gem5 (only the MTBUF/TBUFFER variants do), so this test is
 * expected to abort the simulator with "unimplemented VEGA instruction".  It is
 * provided as a ready self-checking artifact once gem5 implements the
 * instruction; the verification below is exact for a correct identity-format
 * implementation.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o buffer_load_format_d16_xy buffer_load_format_d16_xy.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;

__global__ void
kernel_buffer_load_format_d16_xy(const uint32_t *src, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t base = (uint64_t)src;
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);            /* base lo           */
    srd.y = (uint32_t)((base >> 32) & 0xffffu);        /* base hi, stride=0 */
    srd.z = 0xffffffffu;                               /* num records       */
    /* word3: dstSel identity (X=4,Y=5), numFmt=UINT(4), dataFmt=16_16(11). */
    srd.w = (4u) | (5u << 3) | (4u << 12) | (11u << 15);

    uint32_t voffset = (uint32_t)(i * 4);              /* per-thread record */
    uint32_t data;

    __asm__ volatile(
        "buffer_load_format_d16_xy %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(data)
        : "v"(voffset), "s"(srd)
        : "memory");

    out[i] = data;
}

int
main(void)
{
    uint32_t *h_src = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++)
        h_src[i] = (uint32_t)(i * 2654435761u + 12345u);

    uint32_t *d_src, *d_out;
    GPU_CHECK(hipMalloc(&d_src, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_src, h_src, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_buffer_load_format_d16_xy, dim3(grid), dim3(BLOCK_SIZE),
                       0, 0, d_src, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_src[i]) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], h_src[i]);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_src)); GPU_CHECK(hipFree(d_out));
    free(h_src); free(h_out);
    return pass ? 0 : 1;
}
