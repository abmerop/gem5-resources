/* S_MUL_HI_U32 (ENC_SOP2). SDST=hi32(S0.u32*S1.u32). gem5: sop2.cc. Pattern B (scalar). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint32_t s0,uint32_t s1,uint32_t*out){ uint32_t sd;__asm__ volatile("s_mul_hi_u32 %0, %1, %2\n\t":"=s"(sd):"s"(s0),"s"(s1));if(threadIdx.x==0)out[blockIdx.x]=sd; }
int main(void){
 struct{uint32_t s0,s1;}cs[]={{0,0},{1,1},{0x10000,0x10000},{0xffffffffu,0xffffffffu},{0x12345678u,0x9abcu},{0xdeadbeefu,0x1234u},{65535,65535},{0x80000000u,2},{3,0x55555555u},{0xffff,0xffff}};
 const int NC=sizeof(cs)/sizeof(cs[0]);uint32_t*dout;GPU_CHECK(hipMalloc(&dout,NC*4));
 for(int c=0;c<NC;c++)kern<<<1,BLOCK_SIZE>>>(cs[c].s0,cs[c].s1,dout+c);
 GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 uint32_t h[64];GPU_CHECK(hipMemcpy(h,dout,NC*4,hipMemcpyDeviceToHost));
 int pass=1;for(int c=0;c<NC;c++){uint32_t a=cs[c].s0,b=cs[c].s1,e=((uint32_t)(((uint64_t)a*(uint64_t)b)>>32));if(h[c]!=e){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",c,h[c],e);pass=0;}}
 if(pass)printf("out[4]=%08x\n",h[4]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
