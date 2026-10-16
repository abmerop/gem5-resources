/*
 * HIP test for DS_READ_B96 (ENC_DS, 96-bit LDS read into 3 VGPRs).
 * gem5 (ds.cc): vdst0/1/2[lane] = LDS[addr + 0/4/8].  Each lane writes 3
 * dwords to its own LDS slot, then reads them back via ds_read_b96 and checks
 * all three.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const uint32_t*data,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE*3]; int t=threadIdx.x;
    for(int k=0;k<3;k++) lds[t*3+k]=data[i*3+k];
    __syncthreads();
    __attribute__((address_space(3))) uint32_t*lp=
        (__attribute__((address_space(3))) uint32_t*)(&lds[t*3]);
    uint32_t addr=(uint32_t)(uintptr_t)lp;
    u32x3 v;
    __asm__ volatile("ds_read_b96 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"
                     : "=v"(v) : "v"(addr) : "memory");
    out[i*3+0]=v.x; out[i*3+1]=v.y; out[i*3+2]=v.z;
}
int main(void){
    uint32_t*hd=(uint32_t*)malloc(N*3*4),*ho=(uint32_t*)malloc(N*3*4);
    for(int i=0;i<N*3;i++) hd[i]=(uint32_t)(i*2654435761u+11u);
    uint32_t*dd,*doo;
    GPU_CHECK(hipMalloc(&dd,N*3*4));GPU_CHECK(hipMalloc(&doo,N*3*4));
    GPU_CHECK(hipMemcpy(dd,hd,N*3*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dd,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*3*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N*3;i++){ if(ho[i]!=hd[i]){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,ho[i],hd[i]);pass=0;break;}}
    if(pass)printf("out[0..2]=%08x %08x %08x\n",ho[0],ho[1],ho[2]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
