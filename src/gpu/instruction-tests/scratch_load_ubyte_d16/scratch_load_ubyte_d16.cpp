/*
 * HIP test for SCRATCH_LOAD_UBYTE_D16 (ENC_FLAT_SCRATCH): load a byte into
 * dst[15:0] zero-extended, preserving dst[31:16].  Each lane round-trips a byte
 * through its private scratch slot (scratch_store_byte -> scratch_load_ubyte_d16)
 * and verifies the loaded low half plus the preserved sentinel high half.
 * Decodes to Inst_FLAT__FLAT_LOAD_UBYTE_D16.  Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*in,const int*idx,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    volatile int pad[8]; pad[idx[i]&7]=i;
    int val=(int)(in[i]&0xff); int voff=16;
    uint32_t result=0xC5C50000u; /* high half sentinel must survive */
    __asm__ volatile(
        "scratch_store_byte %1, %2, off offset:0\n\ts_waitcnt vmcnt(0)\n\t"
        "scratch_load_ubyte_d16 %0, %1, off offset:0\n\ts_waitcnt vmcnt(0)\n\t"
        :"+v"(result):"v"(voff),"v"(val):"memory");
    out[i]=result;
}
int main(void){
    uint32_t*hi=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4); int*hx=(int*)malloc(N*4);
    for(int i=0;i<N;i++){hi[i]=(uint32_t)(i*997+3);hx[i]=i;}
    uint32_t*di,*dO; int*dx;
    GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dx,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(di,hi,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dx,hx,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dx,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t exp=0xC5C50000u|(hi[i]&0xffu);
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,ho[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
