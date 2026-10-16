/*
 * HIP test for DS_WRITE2_B64 (ENC_DS, two 64-bit LDS stores at addr+off*8).
 * gem5 (ds.cc): offset0/offset1 are scaled by 8; LDS[addr+off0*8]=DATA0,
 * LDS[addr+off1*8]=DATA1.  With offset0:0 offset1:1 each lane writes two
 * adjacent 64-bit slots.  We read both back to verify.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const uint64_t*d0,const uint64_t*d1,uint64_t*o0,uint64_t*o1,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint64_t lds[2*BLOCK_SIZE]; int t=threadIdx.x;
    lds[2*t]=0; lds[2*t+1]=0; __syncthreads();
    uint64_t a=d0[i], b=d1[i];
    uint32_t addr=(uint32_t)(uintptr_t)(__attribute__((address_space(3)))uint64_t*)(&lds[2*t]);
    __asm__ volatile("ds_write2_b64 %0, %1, %2 offset0:0 offset1:1\n\ts_waitcnt lgkmcnt(0)\n\t"
                     : : "v"(addr), "v"(a), "v"(b) : "memory");
    __syncthreads();
    o0[i]=lds[2*t]; o1[i]=lds[2*t+1];
}
int main(void){
    uint64_t*h0=(uint64_t*)malloc(N*8),*h1=(uint64_t*)malloc(N*8);
    uint64_t*r0=(uint64_t*)malloc(N*8),*r1=(uint64_t*)malloc(N*8);
    for(int i=0;i<N;i++){ h0[i]=0x1111000000000000ull+i; h1[i]=0x2222000000000000ull+(uint64_t)i*7u; }
    uint64_t*d0,*d1,*o0,*o1;
    GPU_CHECK(hipMalloc(&d0,N*8));GPU_CHECK(hipMalloc(&d1,N*8));
    GPU_CHECK(hipMalloc(&o0,N*8));GPU_CHECK(hipMalloc(&o1,N*8));
    GPU_CHECK(hipMemcpy(d0,h0,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d1,h1,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d0,d1,o0,o1,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(r0,o0,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(r1,o1,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++) if(r0[i]!=h0[i]||r1[i]!=h1[i]){
        fprintf(stderr,"FAIL[%d] s0=0x%llx(e 0x%llx) s1=0x%llx(e 0x%llx)\n",i,
            (unsigned long long)r0[i],(unsigned long long)h0[i],
            (unsigned long long)r1[i],(unsigned long long)h1[i]);pass=0;break;}
    if(pass)printf("s0[0]=0x%llx s1[0]=0x%llx\n",(unsigned long long)r0[0],(unsigned long long)r1[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
