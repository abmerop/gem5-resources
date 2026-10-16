/*
 * HIP test for S_BUFFER_ATOMIC_ADD_X2 (ENC_SMEM, 64-bit scalar add atomic,
 * return-old).
 *
 * S_BUFFER_ATOMIC_ADD_X2 performs, atomically on a 64-bit element of a buffer
 * resource, once per wavefront (it is a scalar instruction):
 *
 *   tmp        = MEM[addr]
 *   MEM[addr]  = tmp + DATA            (64-bit add)
 *   SDATA      = tmp                    (old value, with glc return)
 *
 * SMEM buffer instructions address memory through a 128-bit buffer resource
 * descriptor (V#) held in 4 consecutive SGPRs, exactly like the MUBUF
 * s_buffer/buffer family.  Because the instruction is scalar it executes
 * once per wavefront regardless of the number of active lanes, so this test
 * launches a single wavefront (one block of 64 threads) to get exactly one
 * atomic add applied to mem[0].
 *
 * All inputs (the pointer, the add value, the offset) are uniform across the
 * wave (kernel arguments / constants) so the compiler keeps them in SGPRs as
 * the scalar encoding requires.  The sdata operand is read-write ("+s"): it
 * carries the add value in and the returned old value out.
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o s_buffer_atomic_add_x2 \
 *       s_buffer_atomic_add_x2.cpp
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./s_buffer_atomic_add_x2
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
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

static const int BLOCK_SIZE = 64; /* exactly one wavefront on gfx950 */

/*
 * kernel_s_buffer_atomic_add_x2
 *
 * mem[0] += addval (64-bit) via a single scalar SMEM buffer atomic; the
 * returned old value is written to old_out[0].
 */
__global__ void
kernel_s_buffer_atomic_add_x2(uint64_t *mem, uint64_t addval,
                              uint64_t *old_out)
{
    uint64_t base = (uint64_t)mem;
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);          /* base lo            */
    srd.y = (uint32_t)((base >> 32) & 0xffffu);      /* base hi, stride=0  */
    srd.z = 0xffffffffu;                             /* num records (bytes)*/
    srd.w = 0x00020000u;                             /* gfx9 raw buffer    */

    uint64_t d = addval;                             /* add in, old out    */

    __asm__ volatile(
        "s_buffer_atomic_add_x2 %0, %1, 0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+s"(d)
        : "s"(srd)
        : "memory");

    if (threadIdx.x == 0)
        old_out[0] = d;
}

int
main(void)
{
    const uint64_t init   = 0x0000000100000064ULL; /* arbitrary 64-bit val */
    const uint64_t addval = 0x0000000200000007ULL;

    uint64_t h_mem = 0, h_old = 0;

    uint64_t *d_mem, *d_old;
    GPU_CHECK(hipMalloc(&d_mem, sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_old, sizeof(uint64_t)));

    GPU_CHECK(hipMemcpy(d_mem, &init, sizeof(uint64_t),
                        hipMemcpyHostToDevice));

    hipLaunchKernelGGL(kernel_s_buffer_atomic_add_x2,
                       dim3(1), dim3(BLOCK_SIZE), 0, 0,
                       d_mem, addval, d_old);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(&h_mem, d_mem, sizeof(uint64_t),
                        hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(&h_old, d_old, sizeof(uint64_t),
                        hipMemcpyDeviceToHost));

    uint64_t exp_old = init;
    uint64_t exp_mem = init + addval;

    int pass = (h_old == exp_old) && (h_mem == exp_mem);

    if (!pass) {
        fprintf(stderr,
                "FAIL: old=%llx(exp %llx) mem=%llx(exp %llx)\n",
                (unsigned long long)h_old, (unsigned long long)exp_old,
                (unsigned long long)h_mem, (unsigned long long)exp_mem);
    } else {
        printf("old=%llx mem=%llx\n",
               (unsigned long long)h_old, (unsigned long long)h_mem);
    }
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_mem));
    GPU_CHECK(hipFree(d_old));

    return pass ? 0 : 1;
}
