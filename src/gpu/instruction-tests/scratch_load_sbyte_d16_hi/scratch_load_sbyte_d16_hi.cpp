/*
 * HIP test for SCRATCH_LOAD_SBYTE_D16_HI (ENC_FLAT_SCRATCH): load a byte into
 * dst[31:16] sign-extended (to 16 bits), preserving dst[15:0].  Round-trips a
 * signed byte through private scratch; verifies sign-extended high half +
 * preserved sentinel low half.  Decodes to Inst_FLAT__FLAT_LOAD_SBYTE_D16_HI.
 * Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const int8_t*in,const int*idx,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    volatile int pad[8]; pad[idx[i]&7]=i;
    int val=(int)in[i]; int voff=16;
    uint32_t result=0x0000C5C5u; /* low half sentinel must survive */
    __asm__ volatile(
        "scratch_store_byte %1, %2, off offset:0\n\ts_waitcnt vmcnt(0)\n\t"
        "scratch_load_sbyte_d16_hi %0, %1, off offset:0\n\ts_waitcnt vmcnt(0)\n\t"
        :"+v"(result):"v"(voff),"v"(val):"memory");
    out[i]=result;
}
int main(void){
    int8_t*hi=(int8_t*)malloc(N); uint32_t*ho=(uint32_t*)malloc(N*4); int*hx=(int*)malloc(N*4);
    for(int i=0;i<N;i++){hi[i]=(int8_t)(i*37+1);hx[i]=i;}
    int8_t*di; uint32_t*dO; int*dx;
    GPU_CHECK(hipMalloc(&di,N));GPU_CHECK(hipMalloc(&dx,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(di,hi,N,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dx,hx,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dx,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t hh=(uint32_t)(uint16_t)(int16_t)hi[i];
        uint32_t exp=0x0000C5C5u|(hh<<16);
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,ho[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
