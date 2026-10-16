/*
 * HIP test for DS_WRITE2ST64_B32 (ENC_DS, dual LDS store, stride-64).
 * gem5 (ds.cc): writes DATA0 to LDS[addr + OFFSET0*256] and DATA1 to
 * LDS[addr + OFFSET1*256] (OFFSET*4*64 bytes).  With offset0:0 offset1:1 the
 * two dwords land 64 dwords apart.  Each lane owns a private 65-dword region so
 * the two targets never collide with a neighbour.  Read back and verify.
 * Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
#define STRIDE 65
__global__ void kernel(const uint32_t*d0,const uint32_t*d1,uint32_t*o0,uint32_t*o1,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE*STRIDE]; int t=threadIdx.x;
    lds[t*STRIDE]=0xEEEEEEEEu; lds[t*STRIDE+64]=0xEEEEEEEEu; __syncthreads();
    __attribute__((address_space(3))) uint32_t*lp=(__attribute__((address_space(3))) uint32_t*)(&lds[t*STRIDE]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint32_t v0=d0[i],v1=d1[i];
    __asm__ volatile(
        "ds_write2st64_b32 %0, %1, %2 offset0:0 offset1:1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        ::"v"(addr),"v"(v0),"v"(v1):"memory");
    __syncthreads();
    o0[i]=lds[t*STRIDE]; o1[i]=lds[t*STRIDE+64];
}
int main(void){
    uint32_t*h0=(uint32_t*)malloc(N*4),*h1=(uint32_t*)malloc(N*4);
    uint32_t*o0=(uint32_t*)malloc(N*4),*o1=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h0[i]=(uint32_t)(i*2654435761u+1);h1[i]=(uint32_t)(i*40503u+7);}
    uint32_t*d0,*d1,*do0,*do1;
    GPU_CHECK(hipMalloc(&d0,N*4));GPU_CHECK(hipMalloc(&d1,N*4));GPU_CHECK(hipMalloc(&do0,N*4));GPU_CHECK(hipMalloc(&do1,N*4));
    GPU_CHECK(hipMemcpy(d0,h0,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d1,h1,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d0,d1,do0,do1,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(o0,do0,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(o1,do1,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++) if(o0[i]!=h0[i]||o1[i]!=h1[i]){fprintf(stderr,"FAIL[%d] o0=%u(e %u) o1=%u(e %u)\n",i,o0[i],h0[i],o1[i],h1[i]);pass=0;break;}
    if(pass)printf("o0[0]=%u o1[0]=%u\n",o0[0],o1[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
