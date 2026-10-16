/* S_ANDN2_B64 (ENC_SOP2,64-bit). SDST=S0 & ~S1. gem5: sop2.cc. Pattern B (scalar). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint64_t s0,uint64_t s1,uint64_t*out){ uint64_t sd;__asm__ volatile("s_andn2_b64 %0, %1, %2\n\t":"=s"(sd):"s"(s0),"s"(s1));if(threadIdx.x==0)out[blockIdx.x]=sd; }
int main(void){
 struct{uint64_t s0,s1;}cs[]={{0,0},{0xffffffffffffffffull,0},{0xffffffffffffffffull,0xffffffffffffffffull},{0x0123456789abcdefull,0xfedcba9876543210ull},{0xdeadbeefcafebabeull,0x1122334455667788ull},{0xaaaaaaaaaaaaaaaaull,0x5555555555555555ull}};
 const int NC=sizeof(cs)/sizeof(cs[0]);uint64_t*dout;GPU_CHECK(hipMalloc(&dout,NC*8));
 for(int c=0;c<NC;c++)kern<<<1,BLOCK_SIZE>>>(cs[c].s0,cs[c].s1,dout+c);
 GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 uint64_t h[64];GPU_CHECK(hipMemcpy(h,dout,NC*8,hipMemcpyDeviceToHost));
 int pass=1;for(int c=0;c<NC;c++){uint64_t a=cs[c].s0,b=cs[c].s1,e=(a & ~b);if(h[c]!=e){fprintf(stderr,"FAIL[%d] got %016llx exp %016llx\n",c,(unsigned long long)h[c],(unsigned long long)e);pass=0;}}
 if(pass)printf("out[3]=%016llx\n",(unsigned long long)h[3]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
