/*
 * HIP test for V_PK_MAX_U16 (VOP3P packed 2x u16 max).
 * gem5 (vop3p.cc): per 16-bit lane, vdst = (S0 >= S1) ? S0 : S1 (unsigned).
 * Variants: base.
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_pk_max_u16 v_pk_max_u16.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
 "HIP error %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kern(const uint32_t*a,const uint32_t*b,uint32_t*o,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint32_t s0=a[i],s1=b[i],vd;
 __asm__ volatile("v_pk_max_u16 %0,%1,%2\n\t":"=v"(vd):"v"(s0),"v"(s1));o[i]=vd;}

static inline uint32_t pk(uint16_t lo,uint16_t hi){return ((uint32_t)hi<<16)|lo;}

int main(void){
 uint32_t*ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
 for(int i=0;i<N;i++){ha[i]=pk((uint16_t)(i*7+3),(uint16_t)(i*101+5));hb[i]=pk((uint16_t)(i*13+9),(uint16_t)(i*31+1));}
 uint32_t*da,*db,*do_;GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&do_,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,do_,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,do_,N*4,hipMemcpyDeviceToHost));int pass=1;
 for(int i=0;i<N;i++){
  uint16_t al=ha[i]&0xffff,ah=ha[i]>>16,bl=hb[i]&0xffff,bh=hb[i]>>16;
  uint32_t exp=pk(al>=bl?al:bl, ah>=bh?ah:bh);
  if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,ho[i],exp);pass=0;break;}}
 if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",ho[0],N-1,ho[N-1]);
 printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
