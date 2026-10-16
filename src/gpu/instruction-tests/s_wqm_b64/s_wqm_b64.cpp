/*
 * HIP test for S_WQM_B64 (ENC_SOP1, whole-quad-mode on a 64-bit mask),
 * pattern B.
 *
 * gem5 (sop1.cc / inst_util.hh wholeQuadMode): for each 4-bit group (quad) of
 * the 64-bit source, if any bit in the group is set, all 4 bits of that group
 * are set in the result:
 *
 *   for each nibble n: dst_nibble = (src_nibble != 0) ? 0xF : 0x0
 *
 * The source is supplied as a kernel argument (wavefront-uniform, lives in an
 * SGPR pair).  Each wavefront computes WQM of its own input; lane 0 stores it.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_wqm_b64 s_wqm_b64.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NUM_WF=16,BLOCK_SIZE=64;
__global__ void kernel(const uint64_t*in,uint64_t*out){
    int wf=blockIdx.x;
    uint64_t s=in[wf],d;
    __asm__ volatile("s_wqm_b64 %0, %1\n\t":"=s"(d):"s"(s));
    if(threadIdx.x==0) out[wf]=d;
}
static uint64_t wqm(uint64_t v){
    uint64_t r=0;
    for(int n=0;n<16;n++){uint64_t mask=(uint64_t)0xF<<(n*4); if(v&mask) r|=mask;}
    return r;
}
int main(void){
    uint64_t*h_in=(uint64_t*)malloc(NUM_WF*8),*h_out=(uint64_t*)malloc(NUM_WF*8);
    for(int i=0;i<NUM_WF;i++)
        h_in[i]=((uint64_t)(i*0x1248f3a1u)<<32)|(uint64_t)(i*0x9e3779b1u+1u);
    uint64_t*d_in,*d_out;
    GPU_CHECK(hipMalloc(&d_in,NUM_WF*8));GPU_CHECK(hipMalloc(&d_out,NUM_WF*8));
    GPU_CHECK(hipMemcpy(d_in,h_in,NUM_WF*8,hipMemcpyHostToDevice));
    kernel<<<NUM_WF,BLOCK_SIZE>>>(d_in,d_out);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,NUM_WF*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<NUM_WF;i++){uint64_t exp=wqm(h_in[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] in=%llx got %llx exp %llx\n",i,
            (unsigned long long)h_in[i],(unsigned long long)h_out[i],(unsigned long long)exp);pass=0;break;}}
    if(pass)printf("out[1]=%llx out[%d]=%llx\n",(unsigned long long)h_out[1],NUM_WF-1,(unsigned long long)h_out[NUM_WF-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
