/* S_BFE_U64 (ENC_SOP2): SDST = (S0.u64 >> S1[5:0]) & ((1<<S1[22:16])-1).
 * gem5: sop2.cc (mask uses 32-bit int; widths kept <=24). Pattern B (scalar). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint64_t s0,uint32_t s1,uint64_t*out){
 uint64_t sd;__asm__ volatile("s_bfe_u64 %0, %1, %2\n\t":"=s"(sd):"s"(s0),"s"(s1));
 if(threadIdx.x==0)out[blockIdx.x]=sd;}
static uint32_t enc(uint32_t off,uint32_t w){return (off&0x3f)|((w&0x7f)<<16);}
int main(void){
 struct{uint64_t s0;uint32_t off,w;}cs[]={{0x0123456789abcdefull,0,8},{0x0123456789abcdefull,4,8},{0x0123456789abcdefull,8,16},
  {0xffffffffffffffffull,0,24},{0xffffffffffffffffull,10,20},{0xdeadbeefcafebabeull,16,16},
  {0x0000000000000001ull,0,1},{0x8000000000000000ull,32,4},{0xfedcba9876543210ull,1,12},{0x00000000ffffffffull,28,8}};
 const int NC=sizeof(cs)/sizeof(cs[0]);uint64_t*dout;GPU_CHECK(hipMalloc(&dout,NC*8));
 for(int c=0;c<NC;c++)kern<<<1,BLOCK_SIZE>>>(cs[c].s0,enc(cs[c].off,cs[c].w),dout+c);
 GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 uint64_t h[64];GPU_CHECK(hipMemcpy(h,dout,NC*8,hipMemcpyDeviceToHost));
 int pass=1;for(int c=0;c<NC;c++){uint64_t mask=(uint64_t)(((uint32_t)1<<cs[c].w)-1u);uint64_t e=(cs[c].s0>>(cs[c].off&0x3f))&mask;if(h[c]!=e){fprintf(stderr,"FAIL[%d] got %016llx exp %016llx\n",c,(unsigned long long)h[c],(unsigned long long)e);pass=0;}}
 if(pass)printf("out[2]=%016llx\n",(unsigned long long)h[2]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
