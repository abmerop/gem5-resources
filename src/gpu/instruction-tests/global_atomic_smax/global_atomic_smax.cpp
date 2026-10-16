/*
 * HIP test for GLOBAL_ATOMIC_SMAX (ENC_FLAT_GLBL, 32-bit signed atomic max).
 *   old = MEM[addr]; MEM[addr] = max(old, DATA) signed; VDST = old
 * Emitted from atomicMax() on int*. Disjoint slots: mem = max(init,data).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(int*mem,const int*data,int*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    old_out[i]=atomicMax(&mem[i],data[i]);
}
int main(void){
    int*hi=(int*)malloc(N*4),*hd=(int*)malloc(N*4),*hm=(int*)malloc(N*4),*ho=(int*)malloc(N*4);
    for(int i=0;i<N;i++){hi[i]=(i%2? i:-i)*3-5;hd[i]=(i%3? -i:i)*2+1;}
    int*dm,*dd,*dold; GPU_CHECK(hipMalloc(&dm,N*4));GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&dold,N*4));
    GPU_CHECK(hipMemcpy(dm,hi,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,hd,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dm,dd,dold,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(hm,dm,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(ho,dold,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){int eo=hi[i],em=hi[i]>hd[i]?hi[i]:hd[i];
        if(ho[i]!=eo||hm[i]!=em){fprintf(stderr,"FAIL[%d] old=%d(exp %d) mem=%d(exp %d)\n",i,ho[i],eo,hm[i],em);pass=0;break;}}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
