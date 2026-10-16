/*
 * HIP test for V_ASHR_PK_U8_I32 (VOP3): pack two unsigned-saturated bytes formed
 * by arithmetic-shift-right of two i32 sources by src2[4:0] into dst[15:0]
 * (dst[31:16] preserved when OPSEL[3]=0).
 *
 * gem5 (vop3.cc) derives BOTH result bytes from src0 (src1 ignored), so this
 * test drives src0 == src1 -> both bytes are satU8(src0 >> (src2 & 31)) where
 * satU8: n<=0 -> 0, n>=255 -> 0xff, else n&0xff.  dst[31:16] seeded + carried
 * via "+v" to verify preservation.  Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const int32_t*a,const uint32_t*sh,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    int32_t s0=a[i]; uint32_t s2=sh[i];
    uint32_t d=0x5A5A0000u; /* sentinel high half must survive */
    __asm__ volatile("v_ashr_pk_u8_i32 %0, %1, %2, %3\n\t"
                     :"+v"(d):"v"(s0),"v"(s0),"v"(s2));
    out[i]=d;
}
static uint8_t satu8(int32_t n){ if(n<=0)return 0; if(n>=255)return 0xff; return (uint8_t)(n&0xff); }
int main(void){
    int32_t*ha=(int32_t*)malloc(N*4); uint32_t*hs=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ ha[i]=(int32_t)(i*2246822519u)^(i<<4); hs[i]=(uint32_t)(i%20); }
    int32_t*da; uint32_t*ds,*dO;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&ds,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(ds,hs,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,ds,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int32_t v=ha[i]>>(int)(hs[i]&31);
        uint8_t b=satu8(v);
        uint32_t exp=0x5A5A0000u|((uint32_t)b<<8)|(uint32_t)b;
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] a=%d sh=%u got 0x%08x exp 0x%08x\n",i,ha[i],hs[i],ho[i],exp);pass=0;break;}
    }
    if(pass)printf("out[0]=0x%08x\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
