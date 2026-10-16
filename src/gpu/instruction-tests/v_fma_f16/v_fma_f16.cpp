/* HIP test for V_FMA_F16 (VOP3 f16 fused multiply-add) base.
 * gem5: vdst[15:0]=fma(S0,S1,S2)=S0*S1+S2 in f16. Small exact inputs. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP error :0 \n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const _Float16*a,const _Float16*b,const _Float16*c,uint32_t*o,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;_Float16 s0=a[i],s1=b[i],s2=c[i];uint32_t vd;
 __asm__ volatile("v_fma_f16 %0,%1,%2,%3\n\t":"=v"(vd):"v"(s0),"v"(s1),"v"(s2));o[i]=vd;}
int main(void){_Float16*ha=(_Float16*)malloc(N*2),*hb=(_Float16*)malloc(N*2),*hc=(_Float16*)malloc(N*2);uint32_t*ho=(uint32_t*)malloc(N*4);
 for(int i=0;i<N;i++){ha[i]=(_Float16)((float)((i%5)+1));hb[i]=(_Float16)((float)((i%4)+1));hc[i]=(_Float16)((float)((i%7)+1));}
 _Float16*da,*db,*dc;uint32_t*do_;GPU_CHECK(hipMalloc(&da,N*2));GPU_CHECK(hipMalloc(&db,N*2));GPU_CHECK(hipMalloc(&dc,N*2));GPU_CHECK(hipMalloc(&do_,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*2,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*2,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*2,hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,do_,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,do_,N*4,hipMemcpyDeviceToHost));int pass=1;
 for(int i=0;i<N;i++){_Float16 r=(_Float16)((float)ha[i]*(float)hb[i]+(float)hc[i]);uint16_t eb;memcpy(&eb,&r,2);
  if((uint16_t)ho[i]!=eb){fprintf(stderr,"FAIL[%d] got 0x%04x exp 0x%04x\n",i,(uint16_t)ho[i],eb);pass=0;break;}}
 if(pass)printf("out[0]=0x%04x out[%d]=0x%04x\n",(uint16_t)ho[0],N-1,(uint16_t)ho[N-1]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
