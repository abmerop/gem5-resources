/* HIP test for V_MIN_U16 (VOP2 unsigned 16-bit min) base+dpp+sdwa.
 * gem5: vdst=min(src0[15:0],src1[15:0]) unsigned, upper 16 zero. vop2Helper
 * swizzles src0; dpp/sdwa pass src0==src1==a so result is swizzle-direction
 * independent (min commutative). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP error :0 \n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kbase(const uint32_t*a,const uint32_t*b,uint32_t*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint32_t s0=a[i],s1=b[i],vd;__asm__ volatile("v_min_u16_e32 %0,%1,%2\n\t":"=v"(vd):"v"(s0),"v"(s1));o[i]=vd;}
__global__ void kdpp(const uint32_t*a,uint32_t*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint32_t s0=a[i],vd;__asm__ volatile("v_min_u16_dpp %0,%1,%1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(vd):"v"(s0));o[i]=vd;}
__global__ void ksdwa(const uint32_t*a,uint32_t*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint32_t s0=a[i],vd;__asm__ volatile("v_min_u16_sdwa %0,%1,%1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0\n\t":"=v"(vd):"v"(s0));o[i]=vd;}
static inline uint16_t mn(uint32_t x,uint32_t y){uint16_t a=x&0xffff,b=y&0xffff;return a<b?a:b;}
int main(void){uint32_t*ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4),*hba=(uint32_t*)malloc(N*4),*hd=(uint32_t*)malloc(N*4),*hs=(uint32_t*)malloc(N*4);
 for(int i=0;i<N;i++){ha[i]=(uint32_t)(uint16_t)(i*7+3);hb[i]=(uint32_t)(uint16_t)(i*13+9);}
 uint32_t*da,*db,*dba,*dd,*ds;GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dba,N*4));GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&ds,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));int g=N/BLOCK_SIZE;
 hipLaunchKernelGGL(kbase,dim3(g),dim3(BLOCK_SIZE),0,0,da,db,dba,N);GPU_CHECK(hipGetLastError());
 hipLaunchKernelGGL(kdpp,dim3(g),dim3(BLOCK_SIZE),0,0,da,dd,N);GPU_CHECK(hipGetLastError());
 hipLaunchKernelGGL(ksdwa,dim3(g),dim3(BLOCK_SIZE),0,0,da,ds,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(hba,dba,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hd,dd,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hs,ds,N*4,hipMemcpyDeviceToHost));
 static const int perm[4]={1,0,3,2};int pass=1;
 for(int i=0;i<N;i++){uint16_t eb=mn(ha[i],hb[i]);if((uint16_t)hba[i]!=eb){fprintf(stderr,"FAIL base[%d] 0x%04x exp 0x%04x\n",i,(uint16_t)hba[i],eb);pass=0;break;}
  int lane=i%BLOCK_SIZE,wb=i-lane,sl=(lane&~3)+perm[lane&3];uint16_t ed=mn(ha[i],ha[wb+sl]);if((uint16_t)hd[i]!=ed){fprintf(stderr,"FAIL dpp[%d] 0x%04x exp 0x%04x\n",i,(uint16_t)hd[i],ed);pass=0;break;}
  if((uint16_t)hs[i]!=(uint16_t)(ha[i]&0xffff)){fprintf(stderr,"FAIL sdwa[%d]\n",i);pass=0;break;}}
 if(pass)printf("base[1]=0x%04x dpp[1]=0x%04x sdwa[1]=0x%04x\n",(uint16_t)hba[1],(uint16_t)hd[1],(uint16_t)hs[1]);
 printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
