/*
 * HIP test for SCRATCH_STORE_SHORT_D16_HI (ENC_FLAT_SCRATCH): store the high
 * 16 bits (data[31:16]) of a dword to a 2-byte scratch location.  Decodes to
 * Inst_FLAT__FLAT_STORE_SHORT_D16_HI (see gpu_decoder.cc), whose completeAcc
 * writes bits[31:16] of the data operand (the FLAT/scratch d16_hi store path is
 * correct in gem5, unlike the GLOBAL byte_d16_hi path).  Round-trips through
 * private scratch: store data[31:16], then load it back with scratch_load_ushort
 * and verify it equals the high half of the source dword.  Variants: base.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o scratch_store_short_d16_hi scratch_store_short_d16_hi.cpp
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
    uint32_t data=in[i]; int voff=16;
    uint32_t loaded=0;
    __asm__ volatile(
        "scratch_store_short_d16_hi %1, %2, off offset:0\n\ts_waitcnt vmcnt(0)\n\t"
        "scratch_load_ushort %0, %1, off offset:0\n\ts_waitcnt vmcnt(0)\n\t"
        :"=v"(loaded):"v"(voff),"v"(data):"memory");
    out[i]=loaded;
}
int main(void){
    uint32_t*hi=(uint32_t*)malloc(N*4); uint32_t*ho=(uint32_t*)malloc(N*4); int*hx=(int*)malloc(N*4);
    for(int i=0;i<N;i++){hi[i]=(uint32_t)(i*2654435761u+0x12345678u);hx[i]=i;}
    uint32_t*di,*dO; int*dx;
    GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dx,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(di,hi,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dx,hx,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dx,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t exp=(hi[i]>>16)&0xFFFFu;
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x (in 0x%08x)\n",i,ho[i],exp,hi[i]);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x (in 0x%08x)\n",ho[0],hi[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
