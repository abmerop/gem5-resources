/* HIP test for V_FLOOR_F16 (VOP1 f16 floor) base+dpp+sdwa.
 * gem5: result f16 bits = floor(f16(src)) in low 16. Swizzles src0. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP error :0 \n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kbase(const _Float16*a,uint32_t*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;_Float16 s0=a[i];uint32_t vd;__asm__ volatile("v_floor_f16_e32 %0,%1\n\t":"=v"(vd):"v"(s0));o[i]=vd;}
__global__ void kdpp(const _Float16*a,uint32_t*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;_Float16 s0=a[i];uint32_t vd;__asm__ volatile("v_floor_f16_dpp %0,%1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(vd):"v"(s0));o[i]=vd;}
__global__ void ksdwa(const _Float16*a,uint32_t*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;_Float16 s0=a[i];uint32_t vd;__asm__ volatile("v_floor_f16_sdwa %0,%1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0\n\t":"=v"(vd):"v"(s0));o[i]=vd;}
static uint16_t fl(_Float16 x){float f=(float)x;float t=truncf(f);if(f<0.0f&&f!=t)t+=-1.0f;_Float16 h=(_Float16)t;uint16_t b;memcpy(&b,&h,2);return b;}
int main(void){_Float16*ha=(_Float16*)malloc(N*2);uint32_t*hba=(uint32_t*)malloc(N*4),*hd=(uint32_t*)malloc(N*4),*hs=(uint32_t*)malloc(N*4);
 for(int i=0;i<N;i++)ha[i]=(_Float16)(0.25f*(float)((i%41)-20));
 _Float16*da;uint32_t*dba,*dd,*ds;GPU_CHECK(hipMalloc(&da,N*2));GPU_CHECK(hipMalloc(&dba,N*4));GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&ds,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*2,hipMemcpyHostToDevice));int g=N/BLOCK_SIZE;
 hipLaunchKernelGGL(kbase,dim3(g),dim3(BLOCK_SIZE),0,0,da,dba,N);GPU_CHECK(hipGetLastError());
 hipLaunchKernelGGL(kdpp,dim3(g),dim3(BLOCK_SIZE),0,0,da,dd,N);GPU_CHECK(hipGetLastError());
 hipLaunchKernelGGL(ksdwa,dim3(g),dim3(BLOCK_SIZE),0,0,da,ds,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(hba,dba,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hd,dd,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hs,ds,N*4,hipMemcpyDeviceToHost));
 static const int perm[4]={1,0,3,2};int pass=1;
 for(int i=0;i<N;i++){uint16_t eb=fl(ha[i]);if((uint16_t)hba[i]!=eb){fprintf(stderr,"FAIL base[%d] 0x%04x exp 0x%04x\n",i,(uint16_t)hba[i],eb);pass=0;break;}
  int lane=i%BLOCK_SIZE,wb=i-lane,sl=(lane&~3)+perm[lane&3];uint16_t ed=fl(ha[wb+sl]);if((uint16_t)hd[i]!=ed){fprintf(stderr,"FAIL dpp[%d] 0x%04x exp 0x%04x\n",i,(uint16_t)hd[i],ed);pass=0;break;}
  uint16_t es=fl(ha[i]);if((uint16_t)hs[i]!=es){fprintf(stderr,"FAIL sdwa[%d] 0x%04x exp 0x%04x\n",i,(uint16_t)hs[i],es);pass=0;break;}}
 if(pass)printf("base[1]=0x%04x dpp[1]=0x%04x sdwa[1]=0x%04x\n",(uint16_t)hba[1],(uint16_t)hd[1],(uint16_t)hs[1]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
