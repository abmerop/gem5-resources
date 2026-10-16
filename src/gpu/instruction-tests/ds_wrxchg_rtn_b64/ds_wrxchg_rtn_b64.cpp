/* HIP test for DS_WRXCHG_RTN_B64 (ENC_DS LDS atomic, gfx950). Per-lane private
 * LDS slot; host reference matches gem5 AtomicOp functor. base only. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint64_t*init,const uint64_t*data,uint64_t*old_out,uint64_t*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint64_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    __attribute__((address_space(3))) uint64_t*lp=(__attribute__((address_space(3))) uint64_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint64_t v=data[i]; uint64_t old;
    __asm__ volatile("ds_wrxchg_rtn_b64 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    __syncthreads();
    old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    uint64_t*h_i=(uint64_t*)malloc(N*sizeof(uint64_t)),*h_d=(uint64_t*)malloc(N*sizeof(uint64_t)),*h_m=(uint64_t*)malloc(N*sizeof(uint64_t));
    uint64_t*h_old=(uint64_t*)malloc(N*sizeof(uint64_t));
    for(int i=0;i<N;i++){h_i[i]=(uint64_t)((uint64_t)(i*2654435761u)+0x100000000ULL);h_d[i]=(uint64_t)((uint64_t)(i*40503u+7u)*1000003ULL);}
    uint64_t*di,*dd,*dm;
    GPU_CHECK(hipMalloc(&di,N*sizeof(uint64_t)));GPU_CHECK(hipMalloc(&dd,N*sizeof(uint64_t)));GPU_CHECK(hipMalloc(&dm,N*sizeof(uint64_t)));
    uint64_t*d_old;GPU_CHECK(hipMalloc(&d_old,N*sizeof(uint64_t)));
    GPU_CHECK(hipMemcpy(di,h_i,N*sizeof(uint64_t),hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,h_d,N*sizeof(uint64_t),hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,d_old,dm,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_m,dm,N*sizeof(uint64_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,N*sizeof(uint64_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint64_t a=h_i[i],b=h_d[i]; uint64_t em=(b);
        if(h_old[i]!=a){fprintf(stderr,"FAIL[%d] old=%llu(e %llu)\n",i,(long long)h_old[i],(long long)a);pass=0;break;}
        if(h_m[i]!=em){fprintf(stderr,"FAIL[%d] mem=%llu(e %llu)\n",i,(long long)h_m[i],(long long)em);pass=0;break;}}
    if(pass)printf("mem[0]=%llu\n",(long long)h_m[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
