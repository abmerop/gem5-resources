/*
 * HIP test for V_PK_MAD_U16 (VOP3P packed 2x u16 multiply-add).
 * gem5 (vop3p.cc): per 16-bit lane, vdst = clampU16(S0*S1 + S2) (clamp off ->
 * wrapping u16).  Inputs kept small so no wrap ambiguity.  Variants: base.
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_pk_mad_u16 v_pk_mad_u16.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
 "HIP error %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kern(const uint32_t*a,const uint32_t*b,const uint32_t*c,uint32_t*o,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint32_t s0=a[i],s1=b[i],s2=c[i],vd;
 __asm__ volatile("v_pk_mad_u16 %0,%1,%2,%3\n\t":"=v"(vd):"v"(s0),"v"(s1),"v"(s2));o[i]=vd;}

static inline uint32_t pk(uint16_t lo,uint16_t hi){return ((uint32_t)hi<<16)|lo;}

int main(void){
 uint32_t*ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4),*hc=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
 for(int i=0;i<N;i++){
  ha[i]=pk((uint16_t)(i%50),(uint16_t)(i%30));
  hb[i]=pk((uint16_t)(i%20),(uint16_t)(i%15));
  hc[i]=pk((uint16_t)(i%100),(uint16_t)(i%200));}
 uint32_t*da,*db,*dc,*do_;GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&do_,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,do_,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,do_,N*4,hipMemcpyDeviceToHost));int pass=1;
 for(int i=0;i<N;i++){
  uint16_t al=ha[i]&0xffff,ah=ha[i]>>16,bl=hb[i]&0xffff,bh=hb[i]>>16,cl=hc[i]&0xffff,ch=hc[i]>>16;
  uint32_t exp=pk((uint16_t)(al*bl+cl),(uint16_t)(ah*bh+ch));
  if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,ho[i],exp);pass=0;break;}}
 if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",ho[0],N-1,ho[N-1]);
 printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
