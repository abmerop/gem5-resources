/* S_CMP_EQ_I32 (ENC_SOPC). SCC=(S0.i32==S1.i32). gem5: sopc.cc. SCC via S_CSELECT_B32. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(int32_t s0,int32_t s1,uint32_t*out){ uint32_t sd;__asm__ volatile("s_cmp_eq_i32 %1, %2\n\ts_cselect_b32 %0, 1, 0\n\t":"=s"(sd):"s"(s0),"s"(s1));if(threadIdx.x==0)out[blockIdx.x]=sd; }
int main(void){
 struct{int32_t s0,s1;}cs[]={{5,5},{5,3},{3,5},{0,0},{1,2},{2,1},{100,100},{-1,-2},{-2,-1},{2147483647,-2147483648}};
 const int NC=sizeof(cs)/sizeof(cs[0]);uint32_t*dout;GPU_CHECK(hipMalloc(&dout,NC*4));
 for(int c=0;c<NC;c++)kern<<<1,BLOCK_SIZE>>>(cs[c].s0,cs[c].s1,dout+c);
 GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 uint32_t h[64];GPU_CHECK(hipMemcpy(h,dout,NC*4,hipMemcpyDeviceToHost));
 int pass=1;for(int c=0;c<NC;c++){int32_t a=cs[c].s0,b=cs[c].s1;uint32_t e=((a==b))?1u:0u;if(h[c]!=e){fprintf(stderr,"FAIL[%d] got %u exp %u\n",c,h[c],e);pass=0;}}
 if(pass)printf("out[0]=%u out[1]=%u\n",h[0],h[1]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
