/* HIP test for DS_MIN_RTN_I32 (ENC_DS 32-bit LDS signed min, return-old, gfx950).
 * old=LDS[a]; LDS[a]=min(old,DATA) signed; VDST=old. Private slot. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const int*init,const int*data,int*old_out,int*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ int lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    __attribute__((address_space(3))) int*lp=(__attribute__((address_space(3))) int*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; int v=data[i],old;
    __asm__ volatile("ds_min_rtn_i32 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    __syncthreads(); old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    int*hi=(int*)malloc(N*4),*hd=(int*)malloc(N*4),*ho=(int*)malloc(N*4),*hm=(int*)malloc(N*4);
    for(int i=0;i<N;i++){hi[i]=(i%2? i:-i)*3-5;hd[i]=(i%3? -i:i)*2+1;}
    int*di,*dd,*dold,*dm; GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&dold,N*4));GPU_CHECK(hipMalloc(&dm,N*4));
    GPU_CHECK(hipMemcpy(di,hi,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,hd,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,dold,dm,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dold,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hm,dm,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){int eo=hi[i],em=hi[i]<hd[i]?hi[i]:hd[i];
        if(ho[i]!=eo||hm[i]!=em){fprintf(stderr,"FAIL[%d] old=%d(exp %d) mem=%d(exp %d)\n",i,ho[i],eo,hm[i],em);pass=0;break;}}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
