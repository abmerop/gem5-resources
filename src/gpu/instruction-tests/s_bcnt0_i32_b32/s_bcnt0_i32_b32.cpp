/* S_BCNT0_I32_B32 (ENC_SOP1). SDST=count of zero bits in S0. gem5: sop1.cc. Pattern B (scalar). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint32_t s0,uint32_t*out){ uint32_t sd; __asm__ volatile("s_bcnt0_i32_b32 %0, %1\n\t":"=s"(sd):"s"(s0)); if(threadIdx.x==0)out[blockIdx.x]=sd; }
static uint32_t ref(uint32_t a){ return (uint32_t)(32-__builtin_popcount(a)); }
int main(void){
 uint32_t cs[]={0,1,0x80000000u,0xffffffffu,0x00010000u,0x12345678u,0xdeadbeefu,0x0000ff00u,0x00000080u,0x7fffffffu};
 const int NC=sizeof(cs)/sizeof(cs[0]);uint32_t*dout;GPU_CHECK(hipMalloc(&dout,NC*4));
 for(int c=0;c<NC;c++)kern<<<1,BLOCK_SIZE>>>(cs[c],dout+c);
 GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 uint32_t h[64];GPU_CHECK(hipMemcpy(h,dout,NC*4,hipMemcpyDeviceToHost));
 int pass=1;for(int c=0;c<NC;c++){uint32_t e=ref(cs[c]);if(h[c]!=e){fprintf(stderr,"FAIL[%d] in=%08x got %08x exp %08x\n",c,cs[c],h[c],e);pass=0;}}
 if(pass)printf("out[5]=%08x\n",h[5]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
