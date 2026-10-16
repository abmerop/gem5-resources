/*
 * HIP test for FLAT_STORE_BYTE_D16_HI (ENC_FLAT): store bits [23:16] of the
 * data VGPR as a byte to memory:  MEM8[addr] = VDATA[23:16].
 *
 * (The gem5 flat.cc path uses bits(data,23,16); note the GLOBAL variant has a
 * documented byte-select bug, so this test uses the FLAT encoding.)  Each lane
 * builds a dword whose byte[23:16] is a known payload, stores it, and the host
 * verifies the memory byte equals that payload.  Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(uint8_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    /* byte[23:16] = payload; other bytes are filler that must be ignored */
    uint32_t val=((uint32_t)((0x30+i)&0xff)<<16)|0x00000000u; val|=0xAB0000CDu;
    uint64_t addr=(uint64_t)(out+i);
    __asm__ volatile("flat_store_byte_d16_hi %0, %1\n\ts_waitcnt vmcnt(0)\n\t"
                     ::"v"(addr),"v"(val):"memory");
}
int main(void){
    uint8_t*ho=(uint8_t*)malloc(N);
    uint8_t*dO; GPU_CHECK(hipMalloc(&dO,N)); GPU_CHECK(hipMemset(dO,0xAA,N));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint8_t exp=(uint8_t)((0x30+i)&0xff);
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%02x exp 0x%02x\n",i,ho[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%02x out[%d]=0x%02x\n",ho[0],N-1,ho[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
