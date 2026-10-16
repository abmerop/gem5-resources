/*
 * HIP test for V_PK_ADD_U16 (ENC_VOP3P, packed unsigned 16-bit add).
 *   D[15:0] = S0[15:0]+S1[15:0]; D[31:16] = S0[31:16]+S1[31:16] (wrap, no clamp)
 * Pattern A, base only.
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_pk_add_u16 v_pk_add_u16.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*a,const uint32_t*b,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t va=a[i],vb=b[i],vd;
    __asm__ volatile("v_pk_add_u16 %0, %1, %2\n\t":"=v"(vd):"v"(va),"v"(vb));
    out[i]=vd;
}
int main(void){
    uint32_t*h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_a[i]=(uint32_t)(i*131+7)|((uint32_t)(i*977+3)<<16);h_b[i]=(uint32_t)(i*53+11)|((uint32_t)(i*701+19)<<16);}
    uint32_t*da,*db,*dout;GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dout,N*4));
    GPU_CHECK(hipMemcpy(da,h_a,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,h_b,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dout,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,dout,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint16_t lo=(uint16_t)((h_a[i]&0xffff)+(h_b[i]&0xffff));uint16_t hi=(uint16_t)((h_a[i]>>16)+(h_b[i]>>16));
        uint32_t exp=((uint32_t)hi<<16)|lo;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
