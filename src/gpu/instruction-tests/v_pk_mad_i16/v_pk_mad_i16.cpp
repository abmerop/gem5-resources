/*
 * HIP test for V_PK_MAD_I16 (VOP3P, packed 2x i16 multiply-add).
 * gem5 (vop3p.cc): for each 16-bit lane k in {lo,hi}:
 *   D[k] = (i16)(S0[k]*S1[k] + S2[k])   (no clamp).
 * Pattern A (vector), base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const uint32_t*a,const uint32_t*b,const uint32_t*c,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s0=a[i],s1=b[i],s2=c[i],d;
    __asm__ volatile("v_pk_mad_i16 %0, %1, %2, %3\n\t":"=v"(d):"v"(s0),"v"(s1),"v"(s2));
    out[i]=d;
}
static uint32_t ref(uint32_t a,uint32_t b,uint32_t c){
    int16_t a0=(int16_t)(a&0xffff),a1=(int16_t)(a>>16);
    int16_t b0=(int16_t)(b&0xffff),b1=(int16_t)(b>>16);
    int16_t c0=(int16_t)(c&0xffff),c1=(int16_t)(c>>16);
    uint16_t r0=(uint16_t)(int16_t)(a0*b0+c0), r1=(uint16_t)(int16_t)(a1*b1+c1);
    return (uint32_t)r0 | ((uint32_t)r1<<16);
}
int main(void){
    uint32_t*ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4),*hc=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ha[i]=(uint32_t)((i*37-500)&0xffff)|((uint32_t)((i*3-40)&0xffff)<<16);
        hb[i]=(uint32_t)((-i*13+7)&0xffff)|((uint32_t)((i*5-9)&0xffff)<<16);
        hc[i]=(uint32_t)((i*7-1000)&0xffff)|((uint32_t)((i-8)&0xffff)<<16);}
    uint32_t*da,*db,*dc,*dO;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1; for(int i=0;i<N;i++){uint32_t e=ref(ha[i],hb[i],hc[i]);
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,ho[i],e);pass=0;break;}}
    if(pass)printf("out[1]=0x%08x\n",ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
