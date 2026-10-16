/*
 * HIP test for DS_WRXCHG2_RTN_B64 (ENC_DS, dual 64-bit LDS write-exchange,
 * return-old).  Pattern C, base only.  64-bit sibling of DS_WRXCHG2_RTN_B32.
 *
 * gem5 (ds.cc): addr0 = addr + OFFSET0*8 ; addr1 = addr + OFFSET1*8
 *   old0 = LDS64[addr0] ; LDS64[addr0] = DATA0
 *   old1 = LDS64[addr1] ; LDS64[addr1] = DATA1
 *   VDST[0:1]=old0 ; VDST[2:3]=old1   (two consecutive u64 = 4 VGPRs)
 *
 * offset0:0 offset1:1 -> slots 1 u64 apart.  Each lane owns a private 2-u64
 * region so targets never collide.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint64_t u64x2 __attribute__((ext_vector_type(2)));
static const int N=256,BLOCK_SIZE=64;
#define STRIDE 2
__global__ void kernel(const uint64_t*i0,const uint64_t*i1,const uint64_t*d0,const uint64_t*d1,
                       uint64_t*old0_out,uint64_t*old1_out,uint64_t*m0_out,uint64_t*m1_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint64_t lds[BLOCK_SIZE*STRIDE]; int t=threadIdx.x;
    lds[t*STRIDE]=i0[i]; lds[t*STRIDE+1]=i1[i]; __syncthreads();
    __attribute__((address_space(3))) uint64_t*lp=(__attribute__((address_space(3))) uint64_t*)(&lds[t*STRIDE]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint64_t v0=d0[i],v1=d1[i];
    u64x2 old;
    __asm__ volatile(
        "ds_wrxchg2_rtn_b64 %0, %1, %2, %3 offset0:0 offset1:1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"=v"(old):"v"(addr),"v"(v0),"v"(v1):"memory");
    __syncthreads();
    old0_out[i]=old.x; old1_out[i]=old.y;
    m0_out[i]=lds[t*STRIDE]; m1_out[i]=lds[t*STRIDE+1];
}
int main(void){
    uint64_t*hi0=(uint64_t*)malloc(N*8),*hi1=(uint64_t*)malloc(N*8);
    uint64_t*hd0=(uint64_t*)malloc(N*8),*hd1=(uint64_t*)malloc(N*8);
    uint64_t*ho0=(uint64_t*)malloc(N*8),*ho1=(uint64_t*)malloc(N*8);
    uint64_t*hm0=(uint64_t*)malloc(N*8),*hm1=(uint64_t*)malloc(N*8);
    for(int i=0;i<N;i++){hi0[i]=(uint64_t)(i)*0x100000001ull+1;hi1[i]=(uint64_t)(i)*40503ull+7;
                         hd0[i]=0xDEADBEEFCAFEull^((uint64_t)i*13u);hd1[i]=0xC0FFEE1234ull^((uint64_t)i*29u);}
    uint64_t*di0,*di1,*dd0,*dd1,*do0,*do1,*dm0,*dm1;
    GPU_CHECK(hipMalloc(&di0,N*8));GPU_CHECK(hipMalloc(&di1,N*8));
    GPU_CHECK(hipMalloc(&dd0,N*8));GPU_CHECK(hipMalloc(&dd1,N*8));
    GPU_CHECK(hipMalloc(&do0,N*8));GPU_CHECK(hipMalloc(&do1,N*8));
    GPU_CHECK(hipMalloc(&dm0,N*8));GPU_CHECK(hipMalloc(&dm1,N*8));
    GPU_CHECK(hipMemcpy(di0,hi0,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(di1,hi1,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dd0,hd0,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd1,hd1,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di0,di1,dd0,dd1,do0,do1,dm0,dm1,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho0,do0,N*8,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(ho1,do1,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hm0,dm0,N*8,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hm1,dm1,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        if(ho0[i]!=hi0[i]||ho1[i]!=hi1[i]||hm0[i]!=hd0[i]||hm1[i]!=hd1[i]){
            fprintf(stderr,"FAIL[%d] old=(%llu,%llu) mem=(%llu,%llu)\n",i,
                (unsigned long long)ho0[i],(unsigned long long)ho1[i],
                (unsigned long long)hm0[i],(unsigned long long)hm1[i]);pass=0;break;}
    }
    if(pass)printf("old0[0]=%llu mem0[0]=%llu\n",(unsigned long long)ho0[0],(unsigned long long)hm0[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
