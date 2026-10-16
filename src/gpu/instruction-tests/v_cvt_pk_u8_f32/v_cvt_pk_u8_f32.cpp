/*
 * HIP test for V_CVT_PK_U8_F32 (VOP3, convert f32 to u8 and pack into a
 * selected byte of an accumulator dword).
 *
 * Per gem5 src/arch/amdgpu/common/insts/vop3.cc:
 *   sel  = SRC1[1:0]
 *   VDST = (((u8)SRC0.f32) << (8*sel)) | (SRC2 & ~(0xff << (8*sel)))
 *
 * i.e. byte `sel` of the result is the f32 truncated to an unsigned byte; the
 * other three bytes are copied from SRC2.  We exercise all four byte selects
 * with integer-valued f32 inputs (no rounding ambiguity).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_pk_u8_f32 v_cvt_pk_u8_f32.cpp
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

static const int N          = 256;
static const int BLOCK_SIZE = 64;

__global__ void
kernel_v_cvt_pk_u8_f32(const float *f, const uint32_t *sel,
                       const uint32_t *acc, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    float    vf = f[i];
    uint32_t vs = sel[i];
    uint32_t va = acc[i];
    uint32_t vd;

    __asm__ volatile(
        "v_cvt_pk_u8_f32 %0, %1, %2, %3\n\t"
        : "=v"(vd)
        : "v"(vf), "v"(vs), "v"(va));

    out[i] = vd;
}

int
main(void)
{
    float    *h_f   = (float    *)malloc(N * sizeof(float));
    uint32_t *h_sel = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_acc = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    for (int i = 0; i < N; i++) {
        h_f[i]   = (float)(i % 200);          /* integer-valued, fits in u8 */
        h_sel[i] = (uint32_t)(i & 3);         /* byte select 0..3           */
        h_acc[i] = 0xAABBCCDDu ^ (uint32_t)(i * 2654435761u);
    }

    float    *d_f;
    uint32_t *d_sel, *d_acc, *d_out;
    GPU_CHECK(hipMalloc(&d_f,   N * sizeof(float)));
    GPU_CHECK(hipMalloc(&d_sel, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_acc, N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_f,   h_f,   N * sizeof(float),    hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_sel, h_sel, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_acc, h_acc, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_cvt_pk_u8_f32, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_f, d_sel, d_acc, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint32_t sh = 8 * (h_sel[i] & 3);
        uint32_t u8 = (uint32_t)(uint8_t)h_f[i];
        uint32_t expected = (u8 << sh) | (h_acc[i] & ~(0xffu << sh));
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%08x out[%d]=0x%08x\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_f)); GPU_CHECK(hipFree(d_sel));
    GPU_CHECK(hipFree(d_acc)); GPU_CHECK(hipFree(d_out));
    free(h_f); free(h_sel); free(h_acc); free(h_out);
    return pass ? 0 : 1;
}
