/*
 * HIP test for V_MAX_I16 (VOP2 signed 16-bit max), with DPP and SDWA.
 *
 * gem5 (vop2.cc, via vop2Helper -> supports DPP/SDWA):
 *   vdst = (uint16)max((int16)src0[15:0], (int16)src1[15:0]);  upper 16 zeroed.
 *
 *   base : v_max_i16_e32 vd, va, vb            max(a[i], b[i])  (two arrays)
 *   dpp  : both operands = same array a; quad_perm:[1,0,3,2] swizzles src0.
 *          Because max is commutative, max(a[i], a[swizzled]) is independent
 *          of which operand the swizzle is applied to.
 *   sdwa : both operands = a; src0_sel/src1_sel:WORD_0 (identity) -> a[i].
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_max_i16 v_max_i16.cpp
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

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kbase(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint32_t s0 = a[i], s1 = b[i], vd;
    __asm__ volatile("v_max_i16_e32 %0, %1, %2\n\t" : "=v"(vd) : "v"(s0), "v"(s1));
    out[i] = vd;
}
__global__ void
kdpp(const uint32_t *a, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint32_t s0 = a[i], vd;
    __asm__ volatile("v_max_i16_dpp %0, %1, %1 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t" : "=v"(vd) : "v"(s0));
    out[i] = vd;
}
__global__ void
ksdwa(const uint32_t *a, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint32_t s0 = a[i], vd;
    __asm__ volatile("v_max_i16_sdwa %0, %1, %1 dst_sel:DWORD "
                     "dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0\n\t"
                     : "=v"(vd) : "v"(s0));
    out[i] = vd;
}

static inline uint16_t
maxi16(uint32_t x, uint32_t y) {
    int16_t a = (int16_t)(x & 0xffff), b = (int16_t)(y & 0xffff);
    return (uint16_t)(a > b ? a : b);
}

int
main(void) {
    uint32_t *h_a = (uint32_t *)malloc(N * 4), *h_b = (uint32_t *)malloc(N * 4);
    uint32_t *h_base = (uint32_t *)malloc(N * 4);
    uint32_t *h_dpp = (uint32_t *)malloc(N * 4);
    uint32_t *h_sdwa = (uint32_t *)malloc(N * 4);

    for (int i = 0; i < N; i++) {
        h_a[i] = (uint32_t)(uint16_t)(int16_t)((i * 7) % 251 - 125);
        h_b[i] = (uint32_t)(uint16_t)(int16_t)((i * 13) % 241 - 120);
    }

    uint32_t *d_a, *d_b, *d_base, *d_dpp, *d_sdwa;
    GPU_CHECK(hipMalloc(&d_a, N * 4)); GPU_CHECK(hipMalloc(&d_b, N * 4));
    GPU_CHECK(hipMalloc(&d_base, N * 4)); GPU_CHECK(hipMalloc(&d_dpp, N * 4));
    GPU_CHECK(hipMalloc(&d_sdwa, N * 4));
    GPU_CHECK(hipMemcpy(d_a, h_a, N * 4, hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b, h_b, N * 4, hipMemcpyHostToDevice));

    int g = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kbase, dim3(g), dim3(BLOCK_SIZE), 0, 0, d_a, d_b, d_base, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kdpp, dim3(g), dim3(BLOCK_SIZE), 0, 0, d_a, d_dpp, N);
    GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(ksdwa, dim3(g), dim3(BLOCK_SIZE), 0, 0, d_a, d_sdwa, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base, d_base, N * 4, hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp, d_dpp, N * 4, hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa, d_sdwa, N * 4, hipMemcpyDeviceToHost));

    static const int perm[4] = {1, 0, 3, 2};
    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint16_t eb = maxi16(h_a[i], h_b[i]);
        if ((uint16_t)h_base[i] != eb) {
            fprintf(stderr, "FAIL base[%d] got 0x%04x exp 0x%04x\n",
                    i, (uint16_t)h_base[i], eb); pass = 0; break;
        }
        int lane = i % BLOCK_SIZE, wb = i - lane, sl = (lane & ~3) + perm[lane & 3];
        uint16_t ed = maxi16(h_a[i], h_a[wb + sl]);
        if ((uint16_t)h_dpp[i] != ed) {
            fprintf(stderr, "FAIL dpp[%d] got 0x%04x exp 0x%04x\n",
                    i, (uint16_t)h_dpp[i], ed); pass = 0; break;
        }
        uint16_t es = maxi16(h_a[i], h_a[i]);
        if ((uint16_t)h_sdwa[i] != es) {
            fprintf(stderr, "FAIL sdwa[%d] got 0x%04x exp 0x%04x\n",
                    i, (uint16_t)h_sdwa[i], es); pass = 0; break;
        }
    }
    if (pass)
        printf("base[1]=0x%04x dpp[1]=0x%04x sdwa[1]=0x%04x\n",
               (uint16_t)h_base[1], (uint16_t)h_dpp[1], (uint16_t)h_sdwa[1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
