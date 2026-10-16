/*
 * HIP test for V_PK_MAX_F16 (ENC_VOP3P, packed half-precision maximum).
 *
 * V_PK_MAX_F16 operates independently on the two packed f16 halves (per gem5
 * src/arch/amdgpu/common/insts/vop3p.cc, fplibMax = IEEE NaN-suppressing max):
 *
 *   for h in {lo, hi}:  DST[h] = max(SRC0[h], SRC1[h])
 *
 * Inputs are exactly-representable f16 values (no NaN/Inf) so the result is
 * bit-identical to one of the operands and can be verified exactly.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_pk_max_f16 v_pk_max_f16.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./v_pk_max_f16
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
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
kernel_v_pk_max_f16(const uint32_t *a, const uint32_t *b, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t va = a[i];
    uint32_t vb = b[i];
    uint32_t vd;
    __asm__ volatile(
        "v_pk_max_f16 %0, %1, %2\n\t"
        : "=v"(vd)
        : "v"(va), "v"(vb));

    out[i] = vd;
}

static uint16_t
hbits(__half h)
{
    uint16_t b;
    __builtin_memcpy(&b, &h, sizeof(b));
    return b;
}

static uint32_t
pack(__half lo, __half hi)
{
    return ((uint32_t)hbits(hi) << 16) | (uint32_t)hbits(lo);
}

int
main(void)
{
    uint32_t *h_a   = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_b   = (uint32_t *)malloc(N * sizeof(uint32_t));
    uint32_t *h_out = (uint32_t *)malloc(N * sizeof(uint32_t));

    __half *al = (__half *)malloc(N * sizeof(__half));
    __half *ah = (__half *)malloc(N * sizeof(__half));
    __half *bl = (__half *)malloc(N * sizeof(__half));
    __half *bh = (__half *)malloc(N * sizeof(__half));

    for (int i = 0; i < N; i++) {
        al[i] = __float2half((float)(i % 13) * 0.5f);
        ah[i] = __float2half((float)(i % 7) * 0.5f - 1.5f);
        bl[i] = __float2half((float)(i % 5) * 0.25f);
        bh[i] = __float2half((float)(i % 11) * 0.25f - 1.0f);
        h_a[i] = pack(al[i], ah[i]);
        h_b[i] = pack(bl[i], bh[i]);
    }

    uint32_t *d_a, *d_b, *d_out;
    GPU_CHECK(hipMalloc(&d_a,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b,   N * sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(uint32_t)));

    GPU_CHECK(hipMemcpy(d_a, h_a, N * sizeof(uint32_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * sizeof(uint32_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_v_pk_max_f16,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_a, d_b, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        float all = __half2float(al[i]), ahh = __half2float(ah[i]);
        float bll = __half2float(bl[i]), bhh = __half2float(bh[i]);
        uint16_t rlo = hbits(__float2half(all > bll ? all : bll));
        uint16_t rhi = hbits(__float2half(ahh > bhh ? ahh : bhh));
        uint32_t expected = ((uint32_t)rhi << 16) | (uint32_t)rlo;
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%08x expected 0x%08x\n",
                    i, h_out[i], expected);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("out[0]=0x%08x out[%d]=0x%08x\n",
               h_out[0], N - 1, h_out[N - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a)); GPU_CHECK(hipFree(d_b)); GPU_CHECK(hipFree(d_out));
    free(h_a); free(h_b); free(h_out);
    free(al); free(ah); free(bl); free(bh);

    return pass ? 0 : 1;
}
