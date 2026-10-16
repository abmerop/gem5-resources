/*
 * HIP test for BUFFER_ATOMIC_OR_X2 (ENC_MUBUF, 64-bit bitwise-OR atomic,
 * return-old).
 *
 * BUFFER_ATOMIC_OR_X2 performs, atomically on a 64-bit element of a buffer
 * resource:
 *
 *   tmp        = MEM[addr]
 *   MEM[addr]  = tmp | DATA            (bitwise OR, 64-bit)
 *   VDATA      = tmp                    (old value, with sc0 return)
 *
 * MUBUF instructions address memory through a 128-bit buffer resource
 * descriptor (V#) held in 4 consecutive SGPRs.  We build a raw byte buffer
 * descriptor over the target allocation and use a per-thread byte offset
 * (offen).  The vdata operand is read-write ("+v"): it carries the OR value
 * in and the returned old value out.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o buffer_atomic_or_x2 buffer_atomic_or_x2.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./buffer_atomic_or_x2
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
 * kernel_buffer_atomic_or_x2
 *
 * mem[i] = mem[i] | data[i] (64-bit) via a MUBUF buffer atomic; the returned
 * old value is written to old_out[i].
 */
__global__ void
kernel_buffer_atomic_or_x2(uint64_t *mem, const uint64_t *data,
                           uint64_t *old_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    uint64_t base = (uint64_t)mem;
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);          /* base lo            */
    srd.y = (uint32_t)((base >> 32) & 0xffffu);      /* base hi, stride=0  */
    srd.z = 0xffffffffu;                             /* num records (bytes)*/
    srd.w = 0x00020000u;                             /* gfx9 raw buffer    */

    uint32_t voffset = (uint32_t)(i * 8);            /* per-thread 8B slot */
    uint64_t v = data[i];

    __asm__ volatile(
        "buffer_atomic_or_x2 %0, %1, %2, 0 offen sc0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "+v"(v)
        : "v"(voffset), "s"(srd)
        : "memory");

    old_out[i] = v;
}

int
main(void)
{
    uint64_t *h_init = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_data = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_mem  = (uint64_t *)malloc(N * sizeof(uint64_t));
    uint64_t *h_old  = (uint64_t *)malloc(N * sizeof(uint64_t));

    for (int i = 0; i < N; i++) {
        h_init[i] = (uint64_t)(i + 1) * 0x0000000100000001ULL;
        h_data[i] = (uint64_t)(i ^ 0x5a) << 33 | (uint64_t)(i * 7 + 3);
    }

    uint64_t *d_mem, *d_data, *d_old;
    GPU_CHECK(hipMalloc(&d_mem,  N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_data, N * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_old,  N * sizeof(uint64_t)));

    GPU_CHECK(hipMemcpy(d_mem,  h_init, N * sizeof(uint64_t),
                        hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data, h_data, N * sizeof(uint64_t),
                        hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_buffer_atomic_or_x2,
                       dim3(grid), dim3(BLOCK_SIZE), 0, 0,
                       d_mem, d_data, d_old, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_mem, d_mem, N * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old, d_old, N * sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        uint64_t exp_old = h_init[i];
        uint64_t exp_mem = h_init[i] | h_data[i];
        if (h_old[i] != exp_old || h_mem[i] != exp_mem) {
            fprintf(stderr,
                    "FAIL [%d]: old=%llx(exp %llx) mem=%llx(exp %llx)\n",
                    i, (unsigned long long)h_old[i],
                    (unsigned long long)exp_old,
                    (unsigned long long)h_mem[i],
                    (unsigned long long)exp_mem);
            pass = 0;
            break;
        }
    }

    if (pass) {
        printf("old[1]=%llx mem[1]=%llx\n",
               (unsigned long long)h_old[1], (unsigned long long)h_mem[1]);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_mem)); GPU_CHECK(hipFree(d_data));
    GPU_CHECK(hipFree(d_old));
    free(h_init); free(h_data); free(h_mem); free(h_old);

    return pass ? 0 : 1;
}
