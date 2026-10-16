/* HIP test for S_ATOMIC_CMPSWAP (SMEM 32-bit scalar compare-and-swap, gfx950).
 *
 * SDATA is a 2-SGPR pair:
 *   SDATA[0] (low)  : compare value on input; old memory value on output (glc)
 *   SDATA[1] (high) : new value written if the compare matches
 *   tmp = MEM[addr];
 *   if (tmp == cmp) MEM[addr] = new;
 *   if (glc) SDATA[0] = tmp
 *
 * One wavefront per counter (wavefront-uniform).  Even wavefronts use
 * cmp == init so the swap succeeds (mem becomes new); odd wavefronts use
 * cmp != init so it fails (mem stays init).  In both cases old == init.
 * base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NUM_WF=64;

__global__ void
kernel(uint32_t *counters, uint32_t *old_out, const uint32_t *cmp_in,
       const uint32_t *new_in)
{
    int wf = blockIdx.x;
    uint32_t *slot = &counters[wf];
    /* low = compare value, high = new value */
    uint64_t sdata = ((uint64_t)new_in[wf] << 32) | (uint64_t)cmp_in[wf];
    __asm__ volatile(
        "s_atomic_cmpswap %0, %1, 0x0 glc\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "+s"(sdata)
        : "s"(slot)
        : "memory");
    if (threadIdx.x == 0)
        old_out[wf] = (uint32_t)(sdata & 0xFFFFFFFFULL);
}

int
main(void)
{
    uint32_t h_init[NUM_WF], h_mem[NUM_WF], h_old[NUM_WF];
    uint32_t h_cmp[NUM_WF], h_new[NUM_WF];
    for (int w = 0; w < NUM_WF; w++) {
        h_init[w] = (uint32_t)(w * 2654435761u + 123u);
        h_new[w]  = (uint32_t)(w * 40503u + 7u);
        h_cmp[w]  = (w & 1) ? (h_init[w] ^ 0xdeadbeefu) : h_init[w];
    }
    uint32_t *d_c, *d_old, *d_cmp, *d_new;
    GPU_CHECK(hipMalloc(&d_c,   NUM_WF * 4));
    GPU_CHECK(hipMalloc(&d_old, NUM_WF * 4));
    GPU_CHECK(hipMalloc(&d_cmp, NUM_WF * 4));
    GPU_CHECK(hipMalloc(&d_new, NUM_WF * 4));
    GPU_CHECK(hipMemcpy(d_c,   h_init, NUM_WF * 4, hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_cmp, h_cmp,  NUM_WF * 4, hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_new, h_new,  NUM_WF * 4, hipMemcpyHostToDevice));

    kernel<<<NUM_WF, 64>>>(d_c, d_old, d_cmp, d_new);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_mem, d_c,   NUM_WF * 4, hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old, d_old, NUM_WF * 4, hipMemcpyDeviceToHost));

    int pass = 1;
    for (int w = 0; w < NUM_WF; w++) {
        uint32_t em = (w & 1) ? h_init[w] : h_new[w];
        uint32_t eo = h_init[w];
        if (h_old[w] != eo || h_mem[w] != em) {
            fprintf(stderr, "FAIL[%d] old=%u(e %u) mem=%u(e %u)\n",
                    w, h_old[w], eo, h_mem[w], em);
            pass = 0;
            break;
        }
    }
    if (pass)
        printf("old[0]=%u mem[0]=%u mem[1]=%u\n", h_old[0], h_mem[0], h_mem[1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    return pass ? 0 : 1;
}
