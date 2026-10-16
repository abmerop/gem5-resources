/* S_BFE_I32 (ENC_SOP2): signed bitfield extract.
 * gem5: res=(S0>>S1[4:0])&((1<<width)-1); sign-extend from bit(width-1). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint32_t s0,uint32_t s1,uint32_t*out){uint32_t sd;__asm__ volatile("s_bfe_i32 %0, %1, %2\n\t":"=s"(sd):"s"(s0),"s"(s1));if(threadIdx.x==0)out[blockIdx.x]=sd;}
static uint32_t enc(uint32_t off,uint32_t w){return (off&0x1f)|((w&0x7f)<<16);}
static uint32_t ref(uint32_t s0,uint32_t off,uint32_t w){
 uint32_t r=(s0>>(off&0x1f))&(uint32_t)(((uint32_t)1<<w)-1u);
 if((r>>(w-1))&1u) r|=(0xffffffffu<<w);
 return r;}
int main(void){
 struct{uint32_t s0,off,w;}cs[]={{0x000000f0u,4,4},{0x00000080u,7,1},{0xfffffff0u,0,8},{0x12345678u,8,8},
  {0x00000040u,6,1},{0xdeadbeefu,4,12},{0x7fffffffu,0,16},{0x00001000u,12,1},{0xabcdef01u,0,4},{0x80000000u,28,4}};
 const int NC=sizeof(cs)/sizeof(cs[0]);uint32_t*dout;GPU_CHECK(hipMalloc(&dout,NC*4));
 for(int c=0;c<NC;c++)kern<<<1,BLOCK_SIZE>>>(cs[c].s0,enc(cs[c].off,cs[c].w),dout+c);
 GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 uint32_t h[64];GPU_CHECK(hipMemcpy(h,dout,NC*4,hipMemcpyDeviceToHost));
 int pass=1;for(int c=0;c<NC;c++){uint32_t e=ref(cs[c].s0,cs[c].off,cs[c].w);if(h[c]!=e){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",c,h[c],e);pass=0;}}
 if(pass)printf("out[3]=%08x\n",h[3]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
