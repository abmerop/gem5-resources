/*
 * HIP test for S_ANDN2_SAVEEXEC_B64 (SOP1 save EXEC then EXEC = SRC & ~EXEC).
 * gem5 (sop1.cc): sdst = old EXEC; EXEC = src & ~old_EXEC; SCC = (EXEC != 0).
 *
 * The test runs one wavefront, captures (a) the EXEC before, (b) the sdst
 * written by the instruction, and (c) the EXEC after, then restores EXEC.  The
 * scalar moves execute regardless of EXEC, so the restore works even when the
 * instruction zeroes EXEC.  Self-consistent check: sdst == saved and
 * after == (src & ~saved).  Variants: base.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_andn2_saveexec_b64 \
 *            s_andn2_saveexec_b64.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int BLOCK_SIZE = 64;

__global__ void
kern(uint64_t src, uint64_t *out)
{
    uint64_t saved, sdst, after;
    __asm__ volatile(
        "s_mov_b64 %0, exec\n\t"
        "s_andn2_saveexec_b64 %1, %3\n\t"
        "s_mov_b64 %2, exec\n\t"
        "s_mov_b64 exec, %0\n\t"
        : "=s"(saved), "=s"(sdst), "=s"(after) : "s"(src));
    if (threadIdx.x == 0) { out[0]=saved; out[1]=sdst; out[2]=after; }
}

int main(void)
{
    const uint64_t src = 0xF0F0F0F0AAAA5555ull;
    uint64_t *d_out; GPU_CHECK(hipMalloc(&d_out, 3*sizeof(uint64_t)));
    kern<<<1,BLOCK_SIZE>>>(src, d_out);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    uint64_t h[3]; GPU_CHECK(hipMemcpy(h,d_out,3*sizeof(uint64_t),hipMemcpyDeviceToHost));
    uint64_t saved=h[0], sdst=h[1], after=h[2];
    int pass=1;
    if (sdst != saved) { fprintf(stderr,"FAIL sdst 0x%016llx != saved 0x%016llx\n",
        (unsigned long long)sdst,(unsigned long long)saved); pass=0; }
    uint64_t exp_after = src & ~saved;
    if (after != exp_after) { fprintf(stderr,"FAIL after 0x%016llx exp 0x%016llx\n",
        (unsigned long long)after,(unsigned long long)exp_after); pass=0; }
    if (saved == 0) { fprintf(stderr,"FAIL saved exec was 0 (unexpected)\n"); pass=0; }
    if (pass) printf("saved=0x%016llx sdst=0x%016llx after=0x%016llx\n",
        (unsigned long long)saved,(unsigned long long)sdst,(unsigned long long)after);
    printf("%s\n", pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));
    return pass?0:1;
}
