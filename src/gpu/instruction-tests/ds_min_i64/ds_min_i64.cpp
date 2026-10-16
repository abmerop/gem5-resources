/* HIP test for DS_MIN_I64 (ENC_DS LDS atomic, gfx950). Per-lane private
 * LDS slot; host reference matches gem5 AtomicOp functor. base only. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const int64_t*init,const int64_t*data,int64_t*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ int64_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    __attribute__((address_space(3))) int64_t*lp=(__attribute__((address_space(3))) int64_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; int64_t v=data[i]; 
    __asm__ volatile("ds_min_i64 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"::"v"(addr),"v"(v):"memory");
    __syncthreads();
     mem_out[i]=lds[t];
}
int main(void){
    int64_t*h_i=(int64_t*)malloc(N*sizeof(int64_t)),*h_d=(int64_t*)malloc(N*sizeof(int64_t)),*h_m=(int64_t*)malloc(N*sizeof(int64_t));
    
    for(int i=0;i<N;i++){h_i[i]=(int64_t)((int64_t)(i*131-9000)*1000003LL);h_d[i]=(int64_t)((int64_t)(-i*177+5000)*1000003LL);}
    int64_t*di,*dd,*dm;
    GPU_CHECK(hipMalloc(&di,N*sizeof(int64_t)));GPU_CHECK(hipMalloc(&dd,N*sizeof(int64_t)));GPU_CHECK(hipMalloc(&dm,N*sizeof(int64_t)));
    
    GPU_CHECK(hipMemcpy(di,h_i,N*sizeof(int64_t),hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,h_d,N*sizeof(int64_t),hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,dm,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_m,dm,N*sizeof(int64_t),hipMemcpyDeviceToHost));
    
    int pass=1;
    for(int i=0;i<N;i++){ int64_t a=h_i[i],b=h_d[i]; int64_t em=(a<b?a:b);
        
        if(h_m[i]!=em){fprintf(stderr,"FAIL[%d] mem=%lld(e %lld)\n",i,(long long)h_m[i],(long long)em);pass=0;break;}}
    if(pass)printf("mem[0]=%lld\n",(long long)h_m[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
