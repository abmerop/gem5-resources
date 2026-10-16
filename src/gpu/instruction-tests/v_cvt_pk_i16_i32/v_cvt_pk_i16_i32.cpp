/* V_CVT_PK_I16_I32 (ENC_VOP3): D[15:0]=sati16(S0), D[31:16]=sati16(S1).
 * The conversion saturates (clamps) each i32 to [-32768, 32767]; it does
 * NOT truncate. gem5: vop3.cc. Pattern A (vector). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static uint16_t sati16(int32_t v){ v = v>32767?32767:(v<-32768?-32768:v); return (uint16_t)(int16_t)v; }
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const int32_t*a,const int32_t*b,uint32_t*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int32_t s0=a[i],s1=b[i];uint32_t dv;
 __asm__ volatile("v_cvt_pk_i16_i32 %0, %1, %2\n\t":"=v"(dv):"v"(s0),"v"(s1));
 out[i]=dv;}
int main(void){
 int32_t*ha=(int32_t*)malloc(N*4),*hb=(int32_t*)malloc(N*4);uint32_t*ho=(uint32_t*)malloc(N*4);
 for(int i=0;i<N;i++){ha[i]=(int32_t)(i*123457-1000000);hb[i]=(int32_t)((128-i)*98765);}
 int32_t*da,*db;uint32_t*dd;
 GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dd,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dd,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,dd,N*4,hipMemcpyDeviceToHost));
 int pass=1;for(int i=0;i<N;i++){uint32_t e=((uint32_t)sati16(hb[i])<<16)|(uint32_t)sati16(ha[i]);if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,ho[i],e);pass=0;break;}}
 if(pass)printf("out[1]=%08x\n",ho[1]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
