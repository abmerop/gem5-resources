/*
 * HIP test for S_BUFFER_ATOMIC_OR_X2 (ENC_SMEM, 64-bit scalar buffer OR atomic,
 * return-old with glc).
 *
 *   tmp        = MEM[base+offset]
 *   MEM[..]    = tmp | SDATA          (64-bit)
 *   SDATA      = tmp                  (old value, with glc return)
 *
 * Scalar (SMEM) buffer atomic: the memory is addressed through a 4-SGPR buffer
 * resource descriptor (SBASE) plus an immediate offset; SDATA is a read-write
 * SGPR pair carrying the OR operand in and the old value out.  Pattern B
 * (scalar): only lane 0 of the single wavefront performs the atomic, and the
 * descriptor / data are kept in SGPRs.
 *
 * gem5 status note: as of this writing gem5's decoder maps
 * S_BUFFER_ATOMIC_OR_X2 to decodeFatal ("instruction without a class"), so this
 * test is expected to abort the simulator at decode.  It is provided as a ready
 * self-checking artifact once gem5 implements the instruction.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_buffer_atomic_or_x2 s_buffer_atomic_or_x2.cpp
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

static const int BLOCK_SIZE = 64;       /* one wavefront per block */
static const int NUM_WF      = 4;

__global__ void
kernel_s_buffer_atomic_or_x2(uint64_t *mem, uint64_t orval, uint64_t *old_out,
                             uint64_t *new_out)
{
    if (threadIdx.x != 0)
        return;

    /* one 64-bit slot per wavefront */
    uint64_t base = (uint64_t)(mem + blockIdx.x);
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);
    srd.y = (uint32_t)((base >> 32) & 0xffffu);
    srd.z = 0xffffffffu;
    srd.w = 0x00020000u;

    uint64_t data = orval;             /* OR operand in; old value out (glc) */
    __asm__ volatile(
        "s_buffer_atomic_or_x2 %0, %1, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+s"(data)
        : "s"(srd)
        : "memory");

    old_out[blockIdx.x] = data;
    new_out[blockIdx.x] = mem[blockIdx.x];
}

int
main(void)
{
    const uint64_t orval = 0x00f000f0000ff00fULL;

    uint64_t h_init[NUM_WF];
    for (int w = 0; w < NUM_WF; w++)
        h_init[w] = 0x1111222233334444ULL + ((uint64_t)w << 20);

    uint64_t *d_mem, *d_old, *d_new;
    GPU_CHECK(hipMalloc(&d_mem, NUM_WF * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_old, NUM_WF * sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_new, NUM_WF * sizeof(uint64_t)));
    GPU_CHECK(hipMemcpy(d_mem, h_init, NUM_WF * sizeof(uint64_t),
                        hipMemcpyHostToDevice));

    hipLaunchKernelGGL(kernel_s_buffer_atomic_or_x2, dim3(NUM_WF),
                       dim3(BLOCK_SIZE), 0, 0, d_mem, orval, d_old, d_new);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    uint64_t h_old[NUM_WF], h_new[NUM_WF];
    GPU_CHECK(hipMemcpy(h_old, d_old, NUM_WF * sizeof(uint64_t), hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_new, d_new, NUM_WF * sizeof(uint64_t), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int w = 0; w < NUM_WF; w++) {
        uint64_t exp_old = h_init[w];
        uint64_t exp_new = h_init[w] | orval;
        if (h_old[w] != exp_old || h_new[w] != exp_new) {
            fprintf(stderr,
                    "FAIL [%d]: old=0x%016llx(exp 0x%016llx) new=0x%016llx(exp 0x%016llx)\n",
                    w, (unsigned long long)h_old[w], (unsigned long long)exp_old,
                    (unsigned long long)h_new[w], (unsigned long long)exp_new);
            pass = 0; break;
        }
    }

    if (pass)
        printf("old[0]=0x%016llx new[0]=0x%016llx\n",
               (unsigned long long)h_old[0], (unsigned long long)h_new[0]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_mem)); GPU_CHECK(hipFree(d_old)); GPU_CHECK(hipFree(d_new));
    return pass ? 0 : 1;
}
