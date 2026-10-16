/*
 * HIP test for DS_WRXCHG2ST64_RTN_B32 (ENC_DS, dual LDS write-exchange,
 * stride-64, return-old).  Pattern C, base only.
 *
 * gem5 (ds.cc): exchanges two dwords with LDS and returns the old values.
 *   addr0 = addr + OFFSET0*4*64 ; addr1 = addr + OFFSET1*4*64
 *   old0  = LDS[addr0] ; LDS[addr0] = DATA0
 *   old1  = LDS[addr1] ; LDS[addr1] = DATA1
 *   VDST[0]=old0 ; VDST[1]=old1     (two consecutive VGPRs)
 *
 * With offset0:0 offset1:1 the two slots land 64 dwords apart.  Each lane owns
 * a private 65-dword region so targets never collide with a neighbour.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
static const int N=256,BLOCK_SIZE=64;
#define STRIDE 65
__global__ void kernel(const uint32_t*i0,const uint32_t*i1,const uint32_t*d0,const uint32_t*d1,
                       uint32_t*old0_out,uint32_t*old1_out,uint32_t*m0_out,uint32_t*m1_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE*STRIDE]; int t=threadIdx.x;
    lds[t*STRIDE]=i0[i]; lds[t*STRIDE+64]=i1[i]; __syncthreads();
    __attribute__((address_space(3))) uint32_t*lp=(__attribute__((address_space(3))) uint32_t*)(&lds[t*STRIDE]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint32_t v0=d0[i],v1=d1[i];
    u32x2 old;
    __asm__ volatile(
        "ds_wrxchg2st64_rtn_b32 %0, %1, %2, %3 offset0:0 offset1:1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"=v"(old):"v"(addr),"v"(v0),"v"(v1):"memory");
    __syncthreads();
    old0_out[i]=old.x; old1_out[i]=old.y;
    m0_out[i]=lds[t*STRIDE]; m1_out[i]=lds[t*STRIDE+64];
}
int main(void){
    uint32_t*hi0=(uint32_t*)malloc(N*4),*hi1=(uint32_t*)malloc(N*4);
    uint32_t*hd0=(uint32_t*)malloc(N*4),*hd1=(uint32_t*)malloc(N*4);
    uint32_t*ho0=(uint32_t*)malloc(N*4),*ho1=(uint32_t*)malloc(N*4);
    uint32_t*hm0=(uint32_t*)malloc(N*4),*hm1=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){hi0[i]=(uint32_t)(i*2654435761u+1);hi1[i]=(uint32_t)(i*40503u+7);
                         hd0[i]=(uint32_t)(0xDEADBEEFu^(i*13u));hd1[i]=(uint32_t)(0xC0FFEEu^(i*29u));}
    uint32_t*di0,*di1,*dd0,*dd1,*do0,*do1,*dm0,*dm1;
    GPU_CHECK(hipMalloc(&di0,N*4));GPU_CHECK(hipMalloc(&di1,N*4));
    GPU_CHECK(hipMalloc(&dd0,N*4));GPU_CHECK(hipMalloc(&dd1,N*4));
    GPU_CHECK(hipMalloc(&do0,N*4));GPU_CHECK(hipMalloc(&do1,N*4));
    GPU_CHECK(hipMalloc(&dm0,N*4));GPU_CHECK(hipMalloc(&dm1,N*4));
    GPU_CHECK(hipMemcpy(di0,hi0,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(di1,hi1,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dd0,hd0,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dd1,hd1,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di0,di1,dd0,dd1,do0,do1,dm0,dm1,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho0,do0,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(ho1,do1,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hm0,dm0,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hm1,dm1,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        if(ho0[i]!=hi0[i]||ho1[i]!=hi1[i]||hm0[i]!=hd0[i]||hm1[i]!=hd1[i]){
            fprintf(stderr,"FAIL[%d] old=(%u,%u)exp(%u,%u) mem=(%u,%u)exp(%u,%u)\n",
                i,ho0[i],ho1[i],hi0[i],hi1[i],hm0[i],hm1[i],hd0[i],hd1[i]);pass=0;break;}
    }
    if(pass)printf("old0[0]=%u old1[0]=%u mem0[0]=%u mem1[0]=%u\n",ho0[0],ho1[0],hm0[0],hm1[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
