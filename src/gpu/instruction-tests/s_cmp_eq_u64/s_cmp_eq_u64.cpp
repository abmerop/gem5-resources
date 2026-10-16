/* HIP test for S_CMP_EQ_U64 (ENC_SOPC, 64-bit). SCC=(S0.u64==S1.u64).
 * gem5: sopc.cc. SCC captured via S_CSELECT_B32. Pattern B (scalar). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint64_t s0,uint64_t s1,uint32_t*out){
  uint32_t sd; __asm__ volatile("s_cmp_eq_u64 %1, %2\n\ts_cselect_b32 %0, 1, 0\n\t":"=s"(sd):"s"(s0),"s"(s1));
  if(threadIdx.x==0)out[blockIdx.x]=sd;
}
int main(void){
  struct{uint64_t s0,s1;}cs[]={{0,0},{1,1},{1,2},{0xffffffffffffffffull,0xffffffffffffffffull},
    {0x100000000ull,0xffffffffull},{0x0123456789abcdefull,0x0123456789abcdefull},
    {0x0123456789abcdefull,0x0123456789abcde0ull},{0,0xffffffffffffffffull}};
  const int NC=sizeof(cs)/sizeof(cs[0]); uint32_t*dout;GPU_CHECK(hipMalloc(&dout,NC*4));
  for(int c=0;c<NC;c++)kern<<<1,BLOCK_SIZE>>>(cs[c].s0,cs[c].s1,dout+c);
  GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
  uint32_t h[64];GPU_CHECK(hipMemcpy(h,dout,NC*4,hipMemcpyDeviceToHost));
  int pass=1; for(int c=0;c<NC;c++){uint64_t a=cs[c].s0,b=cs[c].s1; uint32_t e=(a==b)?1u:0u; if(h[c]!=e){fprintf(stderr,"FAIL[%d] got %u exp %u\n",c,h[c],e);pass=0;}}
  if(pass)printf("out[0]=%u out[2]=%u\n",h[0],h[2]);
  printf("%s\n",pass?"PASSED":"FAILED"); return pass?0:1;
}
