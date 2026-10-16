/* S_FF1_I32_B64 (ENC_SOP1,64-bit src). SDST=index of lowest set bit; -1 if zero. gem5: sop1.cc. Pattern B (scalar). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint64_t s0,uint32_t*out){ uint32_t sd; __asm__ volatile("s_ff1_i32_b64 %0, %1\n\t":"=s"(sd):"s"(s0)); if(threadIdx.x==0)out[blockIdx.x]=sd; }
static uint32_t ref(uint64_t a){ if(a==0)return 0xffffffffu;return (uint32_t)__builtin_ctzll(a); }
int main(void){
 uint64_t cs[]={0,1,0x8000000000000000ull,0xffffffffffffffffull,0x0000000100000000ull,0x0123456789abcdefull,0xdeadbeefcafebabeull,0x0000ff0000000000ull,0x0000000000000080ull,0x00000000ffffffffull};
 const int NC=sizeof(cs)/sizeof(cs[0]);uint32_t*dout;GPU_CHECK(hipMalloc(&dout,NC*4));
 for(int c=0;c<NC;c++)kern<<<1,BLOCK_SIZE>>>(cs[c],dout+c);
 GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 uint32_t h[64];GPU_CHECK(hipMemcpy(h,dout,NC*4,hipMemcpyDeviceToHost));
 int pass=1;for(int c=0;c<NC;c++){uint32_t e=ref(cs[c]);if(h[c]!=e){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",c,h[c],e);pass=0;}}
 if(pass)printf("out[5]=%08x\n",h[5]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
