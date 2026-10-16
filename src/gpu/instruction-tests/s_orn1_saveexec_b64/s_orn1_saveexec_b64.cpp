/*
 * HIP test for S_ORN1_SAVEEXEC_B64 (SOP1).
 * gem5 (sop1.cc): SDST = EXEC(old); EXEC = ~S0 | EXEC(old); SCC = (EXEC!=0).
 * We set a custom old EXEC inside the asm block (so the OR result actually
 * depends on S0), capture SDST and the new EXEC, then restore EXEC.  All ops
 * are scalar; only lane 0 stores after EXEC is restored.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint64_t src,uint64_t oldv,uint64_t*sdst_out,uint64_t*exec_out){
    uint64_t saved,sdst,newexec;
    __asm__ volatile(
        "s_mov_b64 %0, exec\n\t"
        "s_mov_b64 exec, %4\n\t"
        "s_orn1_saveexec_b64 %1, %3\n\t"
        "s_mov_b64 %2, exec\n\t"
        "s_mov_b64 exec, %0\n\t"
        : "=&s"(saved),"=s"(sdst),"=s"(newexec)
        : "s"(src),"s"(oldv) : "scc");
    if(threadIdx.x==0){ sdst_out[blockIdx.x]=sdst; exec_out[blockIdx.x]=newexec; }
}
int main(void){
    uint64_t src[]={0x0ull,0xffffffffffffffffull,0x00000000ffffffffull,
                    0xaaaaaaaaaaaaaaaaull,0x0123456789abcdefull,0xf0f0f0f0f0f0f0f0ull};
    uint64_t old[]={0xffffffffffffffffull,0x0ull,0xffffffff00000000ull,
                    0x5555555555555555ull,0xdeadbeefcafef00dull,0x0f0f0f0f0f0f0f0full};
    int NC=sizeof(src)/sizeof(src[0]);
    uint64_t*ds,*de; GPU_CHECK(hipMalloc(&ds,NC*8));GPU_CHECK(hipMalloc(&de,NC*8));
    for(int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(src[c],old[c],ds+c,de+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint64_t hs[16],he[16];
    GPU_CHECK(hipMemcpy(hs,ds,NC*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(he,de,NC*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++){ uint64_t es=old[c], ee=(~src[c])|old[c];
        if(hs[c]!=es||he[c]!=ee){fprintf(stderr,"FAIL[%d] sdst=0x%llx(e 0x%llx) exec=0x%llx(e 0x%llx)\n",
            c,(unsigned long long)hs[c],(unsigned long long)es,(unsigned long long)he[c],(unsigned long long)ee);pass=0;}}
    if(pass)printf("sdst[0]=0x%llx exec[0]=0x%llx\n",(unsigned long long)hs[0],(unsigned long long)he[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
