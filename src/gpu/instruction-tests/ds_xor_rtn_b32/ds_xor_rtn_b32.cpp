/* HIP test for DS_XOR_RTN_B32 (ENC_DS LDS atomic, gfx950). Per-lane private
 * LDS slot; host reference matches gem5 AtomicOp functor. base only. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*init,const uint32_t*data,uint32_t*old_out,uint32_t*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    __attribute__((address_space(3))) uint32_t*lp=(__attribute__((address_space(3))) uint32_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint32_t v=data[i]; uint32_t old;
    __asm__ volatile("ds_xor_rtn_b32 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t":"=v"(old):"v"(addr),"v"(v):"memory");
    __syncthreads();
    old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    uint32_t*h_i=(uint32_t*)malloc(N*sizeof(uint32_t)),*h_d=(uint32_t*)malloc(N*sizeof(uint32_t)),*h_m=(uint32_t*)malloc(N*sizeof(uint32_t));
    uint32_t*h_old=(uint32_t*)malloc(N*sizeof(uint32_t));
    for(int i=0;i<N;i++){h_i[i]=(uint32_t)(i*2654435761u+123u);h_d[i]=(uint32_t)(i*40503u+7u);}
    uint32_t*di,*dd,*dm;
    GPU_CHECK(hipMalloc(&di,N*sizeof(uint32_t)));GPU_CHECK(hipMalloc(&dd,N*sizeof(uint32_t)));GPU_CHECK(hipMalloc(&dm,N*sizeof(uint32_t)));
    uint32_t*d_old;GPU_CHECK(hipMalloc(&d_old,N*sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(di,h_i,N*sizeof(uint32_t),hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd,h_d,N*sizeof(uint32_t),hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,d_old,dm,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_m,dm,N*sizeof(uint32_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,N*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint32_t a=h_i[i],b=h_d[i]; uint32_t em=(a ^ b);
        if(h_old[i]!=a){fprintf(stderr,"FAIL[%d] old=%u(e %u)\n",i,h_old[i],a);pass=0;break;}
        if(h_m[i]!=em){fprintf(stderr,"FAIL[%d] mem=%u(e %u)\n",i,h_m[i],em);pass=0;break;}}
    if(pass)printf("mem[0]=%u\n",h_m[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
