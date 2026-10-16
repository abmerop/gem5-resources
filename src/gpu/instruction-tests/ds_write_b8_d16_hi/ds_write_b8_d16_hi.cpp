/*
 * HIP test for DS_WRITE_B8_D16_HI (ENC_DS): store bits [23:16] of the data VGPR
 * as a byte to LDS:  LDS8[addr] = DATA[23:16].
 *
 * Each lane owns a distinct LDS byte slot (pre-filled with a sentinel), writes
 * byte [23:16] of a known dword, reads it back, and verifies it equals
 * DATA[23:16].  Variants exercised: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const uint32_t*data,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint8_t lds[BLOCK_SIZE];
    int t=threadIdx.x;
    lds[t]=0xEE; __syncthreads();
    __attribute__((address_space(3))) uint8_t *lp=
        (__attribute__((address_space(3))) uint8_t *)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp;
    uint32_t vdata=data[i];
    __asm__ volatile("ds_write_b8_d16_hi %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"
                     ::"v"(addr),"v"(vdata):"memory");
    __syncthreads();
    out[i]=(uint32_t)lds[t];
}
int main(void){
    uint32_t*hd=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) hd[i]=(uint32_t)(i*277u+13u)*0x1010101u;
    uint32_t*dd,*dO; GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(dd,hd,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dd,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t exp=(hd[i]>>16)&0xffu;
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%02x exp 0x%02x\n",i,ho[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%02x\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
