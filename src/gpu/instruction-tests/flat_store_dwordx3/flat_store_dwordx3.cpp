/*
 * HIP test for FLAT_STORE_DWORDX3 (ENC_FLAT, 96-bit / 3-dword flat store).
 *
 * FLAT_STORE_DWORDX3 stores three consecutive 32-bit dwords from a VGPR
 * triple to the 64-bit flat address held in a VGPR pair:
 *
 *   FLAT[addr + 0] = DATA.x
 *   FLAT[addr + 4] = DATA.y
 *   FLAT[addr + 8] = DATA.z
 *
 * The instruction is forced with inline assembly.  The data operand is a
 * 3-element vector (ext_vector_type(3)) so the compiler allocates three
 * consecutive VGPRs; the address operand is the flat pointer cast to a 64-bit
 * VGPR pair.  s_waitcnt vmcnt(0) ensures the store has issued before the host
 * reads back.
 *
 * Each thread owns a disjoint 3-dword region of the output buffer, so the
 * stored values are deterministic and verified directly against memory.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o flat_store_dwordx3 flat_store_dwordx3.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./flat_store_dwordx3
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

typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));

static const int N          = 256;
static const int BLOCK_SIZE = 64;

/*
 * kernel_flat_store_dwordx3
 *
 * Thread i writes {a, a+1, a+2} (a = i*100 + 7) to out[i*3 .. i*3+2] using a
 * single 96-bit flat store.
 */
__global__ void
kernel_flat_store_dwordx3(uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint32_t base = (uint32_t)(i * 100 + 7);
    u32x3 data;
    data.x = base;
    data.y = base + 1;
    data.z = base + 2;

    uint64_t addr = (uint64_t)(out + (size_t)i * 3);

    __asm__ volatile(
        "flat_store_dwordx3 %0, %1\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :
        : "v"(addr), "v"(data)
        : "memory");
}

int
main(void)
{
    const int total = N * 3;
    uint32_t *h_out = (uint32_t *)malloc(total * sizeof(uint32_t));

    uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_out, total * sizeof(uint32_t)));
    GPU_CHECK(hipMemset(d_out, 0xff, total * sizeof(uint32_t)));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_flat_store_dwordx3,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, total * sizeof(uint32_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        uint32_t base = (uint32_t)(i * 100 + 7);
        for (int k = 0; k < 3; k++) {
            uint32_t expected = base + k;
            if (h_out[i * 3 + k] != expected) {
                fprintf(stderr, "FAIL out[%d]: got %u expected %u\n",
                        i * 3 + k, h_out[i * 3 + k], expected);
                pass = 0;
                break;
            }
        }
    }

    if (pass) {
        printf("out[0..2]=%u,%u,%u out[%d]=%u\n",
               h_out[0], h_out[1], h_out[2], total - 1, h_out[total - 1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_out));
    free(h_out);

    return pass ? 0 : 1;
}
