/*
 * HIP test for S_CSELECT_B64 (ENC_SOP2, conditional select on SCC), pattern B.
 *
 * gem5 (sop2.cc): SDST = SCC ? SSRC0 : SSRC1   (64-bit)
 *
 * SCC is established immediately before the select by an s_cmp inside the same
 * inline-asm block (so the compiler cannot clobber SCC in between).  We use
 * s_cmp_eq_u32 on two scalar kernel args to set SCC deterministically, then
 * select between two 64-bit values.  Both true and false cases are exercised.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_cselect_b64 s_cselect_b64.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NUM_WF=16,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*ca,const uint32_t*cb,
                       const uint64_t*x0,const uint64_t*x1,uint64_t*out){
    int wf=blockIdx.x;
    uint32_t a=ca[wf],b=cb[wf];
    uint64_t s0=x0[wf],s1=x1[wf],d;
    __asm__ volatile(
        "s_cmp_eq_u32 %1, %2\n\t"      /* SCC = (a == b) */
        "s_cselect_b64 %0, %3, %4\n\t" /* d = SCC ? s0 : s1 */
        :"=s"(d):"s"(a),"s"(b),"s"(s0),"s"(s1):"scc");
    if(threadIdx.x==0) out[wf]=d;
}
int main(void){
    uint32_t*h_ca=(uint32_t*)malloc(NUM_WF*4),*h_cb=(uint32_t*)malloc(NUM_WF*4);
    uint64_t*h_x0=(uint64_t*)malloc(NUM_WF*8),*h_x1=(uint64_t*)malloc(NUM_WF*8),*h_out=(uint64_t*)malloc(NUM_WF*8);
    for(int i=0;i<NUM_WF;i++){
        h_ca[i]=i; h_cb[i]=(i&1)?i:i+1;   /* equal when i even -> SCC true */
        h_x0[i]=0xAAAA000000000000ULL|(uint64_t)i;
        h_x1[i]=0xBBBB000000000000ULL|(uint64_t)i;
    }
    uint32_t*d_ca,*d_cb; uint64_t*d_x0,*d_x1,*d_out;
    GPU_CHECK(hipMalloc(&d_ca,NUM_WF*4));GPU_CHECK(hipMalloc(&d_cb,NUM_WF*4));
    GPU_CHECK(hipMalloc(&d_x0,NUM_WF*8));GPU_CHECK(hipMalloc(&d_x1,NUM_WF*8));GPU_CHECK(hipMalloc(&d_out,NUM_WF*8));
    GPU_CHECK(hipMemcpy(d_ca,h_ca,NUM_WF*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_cb,h_cb,NUM_WF*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_x0,h_x0,NUM_WF*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_x1,h_x1,NUM_WF*8,hipMemcpyHostToDevice));
    kernel<<<NUM_WF,BLOCK_SIZE>>>(d_ca,d_cb,d_x0,d_x1,d_out);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,NUM_WF*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<NUM_WF;i++){
        uint64_t exp=(h_ca[i]==h_cb[i])?h_x0[i]:h_x1[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %llx exp %llx\n",i,
            (unsigned long long)h_out[i],(unsigned long long)exp);pass=0;break;}}
    if(pass)printf("out[0]=%llx out[1]=%llx\n",(unsigned long long)h_out[0],(unsigned long long)h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
