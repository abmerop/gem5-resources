/*
 * HIP test for S_ANDN1_WREXEC_B64 (SOP1).
 * gem5 (instructions.cc): EXEC = ~SSRC0 & EXEC_old; SDST = EXEC (the NEW value,
 * unlike the saveexec forms which return the old EXEC); SCC = (EXEC != 0).
 * All 64 lanes active -> EXEC_old = ~0, so both SDST and new EXEC = ~src.
 * Capture new EXEC then restore via scalar s_mov.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint64_t src, uint64_t*sdst_out, uint64_t*exec_out){
    uint64_t saved, sdst, newexec;
    __asm__ volatile(
        "s_mov_b64 %0, exec\n\t"
        "s_andn1_wrexec_b64 %1, %3\n\t"
        "s_mov_b64 %2, exec\n\t"
        "s_mov_b64 exec, %0\n\t"
        : "=&s"(saved), "=s"(sdst), "=s"(newexec)
        : "s"(src) : "scc");
    if(threadIdx.x==0){ sdst_out[blockIdx.x]=sdst; exec_out[blockIdx.x]=newexec; }
}
int main(void){
    uint64_t cases[]={0x0ull,0xffffffffffffffffull,0x123456789abcdef0ull,
                      0xaaaaaaaaaaaaaaaaull,0xf0f0f0f0f0f0f0f0ull,0x1ull};
    int NC=sizeof(cases)/sizeof(cases[0]);
    uint64_t*ds,*de; GPU_CHECK(hipMalloc(&ds,NC*8)); GPU_CHECK(hipMalloc(&de,NC*8));
    for(int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(cases[c],ds+c,de+c);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    uint64_t hs[16],he[16];
    GPU_CHECK(hipMemcpy(hs,ds,NC*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(he,de,NC*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++){
        uint64_t old=0xffffffffffffffffull, e=~cases[c] & old;
        if(hs[c]!=e||he[c]!=e){fprintf(stderr,"FAIL[%d] sdst=0x%llx exec=0x%llx (e 0x%llx)\n",
            c,(unsigned long long)hs[c],(unsigned long long)he[c],(unsigned long long)e);pass=0;}
    }
    if(pass)printf("sdst[0]=0x%llx exec[0]=0x%llx\n",(unsigned long long)hs[0],(unsigned long long)he[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
