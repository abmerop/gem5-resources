/* HIP test for DS_MAX_RTN_F64 (ENC_DS 64-bit LDS float max, return-old, gfx950).
 * old=LDS[a]; LDS[a]=max(old,DATA); VDST=old. Private slot; distinct finite vals. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const double*init,const double*data,double*old_out,double*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ double lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    __attribute__((address_space(3))) double*lp=(__attribute__((address_space(3))) double*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; double v=data[i],old;
    __asm__ volatile("ds_max_rtn_f64 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    __syncthreads(); old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    double*hi=(double*)malloc(N*8),*hd=(double*)malloc(N*8),*ho=(double*)malloc(N*8),*hm=(double*)malloc(N*8);
    for(int i=0;i<N;i++){hi[i]=(double)((i*37)%211)-105.0;hd[i]=(double)((i*53)%199)-99.0+0.5;}
    double*di,*dd,*dold,*dm; GPU_CHECK(hipMalloc(&di,N*8));GPU_CHECK(hipMalloc(&dd,N*8));GPU_CHECK(hipMalloc(&dold,N*8));GPU_CHECK(hipMalloc(&dm,N*8));
    GPU_CHECK(hipMemcpy(di,hi,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,hd,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,dold,dm,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dold,N*8,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hm,dm,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){double eo=hi[i],em=hi[i]>hd[i]?hi[i]:hd[i];
        if(ho[i]!=eo||hm[i]!=em){fprintf(stderr,"FAIL[%d] old=%g(exp %g) mem=%g(exp %g)\n",i,ho[i],eo,hm[i],em);pass=0;break;}}
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
