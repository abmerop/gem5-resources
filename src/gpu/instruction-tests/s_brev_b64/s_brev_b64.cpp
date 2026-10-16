/* S_BREV_B64 (ENC_SOP1). SDST.u64 = reverseBits(S0.u64). gem5: sop1.cc.
 * Pattern B (scalar): operand passed as kernel arg -> SGPR pair. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(uint64_t s0,uint64_t*out){ uint64_t sd; __asm__ volatile("s_brev_b64 %0, %1\n\t":"=s"(sd):"s"(s0)); if(threadIdx.x==0)out[blockIdx.x]=sd; }
static uint64_t ref(uint64_t a){ uint64_t r=0;for(int b=0;b<64;b++){r=(r<<1)|(a&1ULL);a>>=1;}return r; }
int main(void){
 uint64_t cs[]={0ULL,1ULL,0x8000000000000000ULL,0xffffffffffffffffULL,0x0000000100000000ULL,0x123456789abcdef0ULL,0xdeadbeefcafebabeULL,0x00000000000000ffULL,0x00000000000000ffULL,0x7fffffffffffffffULL};
 const int NC=sizeof(cs)/sizeof(cs[0]);uint64_t*dout;GPU_CHECK(hipMalloc(&dout,NC*8));
 for(int c=0;c<NC;c++)kern<<<1,BLOCK_SIZE>>>(cs[c],dout+c);
 GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 uint64_t h[64];GPU_CHECK(hipMemcpy(h,dout,NC*8,hipMemcpyDeviceToHost));
 int pass=1;for(int c=0;c<NC;c++){uint64_t e=ref(cs[c]);if(h[c]!=e){fprintf(stderr,"FAIL[%d] in=%016llx got %016llx exp %016llx\n",c,(unsigned long long)cs[c],(unsigned long long)h[c],(unsigned long long)e);pass=0;}}
 if(pass)printf("out[5]=%016llx\n",(unsigned long long)h[5]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
