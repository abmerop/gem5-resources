/*
 * HIP test for S_NAND_SAVEEXEC_B64 (ENC_SOP1, save EXEC then NAND into EXEC).
 *
 *   SDST = EXEC; EXEC = ~(S0 & EXEC); SCC = (EXEC != 0)
 *
 * Each block is one full wavefront (64 active lanes), so EXEC = 0xFFFF...F on
 * entry.  The kernel saves EXEC, runs the instruction (capturing SDST and the
 * new EXEC), then restores EXEC.  Pattern B (scalar), base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_nand_saveexec_b64 s_nand_saveexec_b64.cpp
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
kernel_s_nand_saveexec_b64(uint64_t s0, uint64_t *sdst_out, uint64_t *exec_out)
{
    uint64_t sdst, newexec, saved;
    __asm__ volatile(
        "s_mov_b64 %2, exec\n\t"
        "s_nand_saveexec_b64 %0, %3\n\t"
        "s_mov_b64 %1, exec\n\t"
        "s_mov_b64 exec, %2\n\t"
        : "=s"(sdst), "=s"(newexec), "=s"(saved) : "s"(s0));
    if (threadIdx.x == 0) {
        sdst_out[blockIdx.x] = sdst;
        exec_out[blockIdx.x] = newexec;
    }
}

int main(void)
{
    uint64_t cases[] = {
        0x0000000000000000ULL, 0xffffffffffffffffULL, 0x00000000ffffffffULL,
        0xaaaaaaaaaaaaaaaaULL, 0x0123456789abcdefULL, 0xdeadbeefcafef00dULL,
    };
    const int nc=(int)(sizeof(cases)/sizeof(cases[0]));
    uint64_t *d_sdst,*d_exec;
    GPU_CHECK(hipMalloc(&d_sdst,nc*sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_exec,nc*sizeof(uint64_t)));
    for(int c=0;c<nc;c++){
        kernel_s_nand_saveexec_b64<<<1,BLOCK_SIZE>>>(cases[c],d_sdst+c,d_exec+c);
        GPU_CHECK(hipGetLastError());
    }
    GPU_CHECK(hipDeviceSynchronize());
    uint64_t *h_sdst=(uint64_t*)malloc(nc*sizeof(uint64_t));
    uint64_t *h_exec=(uint64_t*)malloc(nc*sizeof(uint64_t));
    GPU_CHECK(hipMemcpy(h_sdst,d_sdst,nc*sizeof(uint64_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_exec,d_exec,nc*sizeof(uint64_t),hipMemcpyDeviceToHost));
    const uint64_t old_exec=0xffffffffffffffffULL;
    int pass=1;
    for(int c=0;c<nc;c++){
        uint64_t ee=~(cases[c]&old_exec);
        if(h_sdst[c]!=old_exec||h_exec[c]!=ee){
            fprintf(stderr,"FAIL case %d: sdst 0x%llx(exp 0x%llx) exec 0x%llx(exp 0x%llx)\n",
                c,(unsigned long long)h_sdst[c],(unsigned long long)old_exec,
                (unsigned long long)h_exec[c],(unsigned long long)ee);pass=0;break;
        }
    }
    if(pass)printf("sdst[0]=0x%llx exec[0]=0x%llx\n",
        (unsigned long long)h_sdst[0],(unsigned long long)h_exec[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
