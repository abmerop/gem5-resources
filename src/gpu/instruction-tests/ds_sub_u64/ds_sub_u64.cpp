/*
 * HIP test for DS_SUB_U64 (ENC_DS, 64-bit LDS subtract, no return).
 *
 * DS_SUB_U64 computes, on a 64-bit LDS location:
 *
 *   LDS[addr] = LDS[addr] - DATA      (64-bit, no return)
 *
 * Each thread owns a distinct 8-byte LDS slot, so there is no contention.  The
 * slot is initialised to init[i], the instruction performs init[i] - data[i],
 * and the post-op slot value is read back and verified (mod 2^64).
 *
 * The 32-bit LDS byte offset is obtained by casting the __shared__ pointer to
 * the local (address_space(3)) space (same approach as ds_rsub_u32).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_sub_u64 ds_sub_u64.cpp
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
kernel_ds_sub_u64(const uint64_t *init, const uint64_t *data, uint64_t *out,
                  int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ uint64_t lds[BLOCK_SIZE];
    int t = threadIdx.x;

    lds[t] = init[i];
    __syncthreads();

    __attribute__((address_space(3))) uint64_t *lp =
        (__attribute__((address_space(3))) uint64_t *)(&lds[t]);
    uint32_t addr  = (uint32_t)(uintptr_t)lp;
    uint64_t vdata = data[i];

    __asm__ volatile(
        "ds_sub_u64 %0, %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :
        : "v"(addr), "v"(vdata)
        : "memory");

    __syncthreads();
    out[i] = lds[t];
}

int
main(void)
{
    uint64_t *h_init = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_data = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_out  = (uint64_t *)malloc(N * sizeof(uint64_t));

    for (int i = 0; i < N; i++) {
        h_init[i] = (uint64_t)(i * 1234567891ull + 0x100000000ull);
        h_data[i] = (uint64_t)(0xfeedface00001234ull - (uint64_t)i * 13);
    }

    uint64_t *d_init, *d_data, *d_out;
    GPU_CHECK(hipMalloc(&d_init, N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_data, N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_out,  N * sizeof(uint64_t)));

    GPU_CHECK(hipMemcpy(d_init, h_init, N * sizeof(uint64_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data, h_data, N * sizeof(uint64_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_ds_sub_u64, dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_init, d_data, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(uint64_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint64_t expected = h_init[i] - h_data[i];
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got 0x%016llx expected 0x%016llx\n",
                    i, (unsigned long long)h_out[i],
                    (unsigned long long)expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=0x%016llx out[%d]=0x%016llx\n",
               (unsigned long long)h_out[0], N - 1,
               (unsigned long long)h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_init)); GPU_CHECK(hipFree(d_data)); GPU_CHECK(hipFree(d_out));
    free(h_init); free(h_data); free(h_out);
    return pass ? 0 : 1;
}
