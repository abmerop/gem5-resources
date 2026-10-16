/*
 * HIP test for GLOBAL_STORE_SHORT_D16_HI (ENC_FLAT_GLBL, store high 16 bits of
 * a VGPR to a 16-bit memory location).  Decodes in gem5 to
 * Inst_FLAT__FLAT_STORE_SHORT_D16_HI.
 *
 *   MEM16[addr] = VDATA[31:16]
 *
 * Thread i builds a 32-bit value whose high half is a known payload (low half
 * is filler that must be ignored), stores its high half, and the host verifies
 * the 16-bit memory word equals the payload.  Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(uint16_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t val=((uint32_t)(0x3000+i*5)<<16)|(uint32_t)0xBEEF; /* hi=payload lo=filler */
    uint64_t addr=(uint64_t)(uintptr_t)(out+i);
    __asm__ volatile("global_store_short_d16_hi %0, %1, off\n\ts_waitcnt vmcnt(0)\n\t"
                     ::"v"(addr),"v"(val):"memory");
}
int main(void){
    uint16_t*h=(uint16_t*)malloc(N*2); uint16_t*d;
    GPU_CHECK(hipMalloc(&d,N*2)); GPU_CHECK(hipMemset(d,0xff,N*2));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h,d,N*2,hipMemcpyDeviceToHost));
    int pass=1; for(int i=0;i<N;i++){uint16_t e=(uint16_t)(0x3000+i*5);
        if(h[i]!=e){fprintf(stderr,"FAIL[%d] got 0x%04x exp 0x%04x\n",i,h[i],e);pass=0;break;}}
    if(pass)printf("out[0]=0x%04x out[%d]=0x%04x\n",h[0],N-1,h[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
