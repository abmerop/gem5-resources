/* HIP test for V_MED3_I32 (ENC_VOP3, signed 32-bit median-of-three).
 * gem5 (vop3.cc): D.i32 = median(S0,S1,S2). Pattern A (vector). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const int32_t*a,const int32_t*b,const int32_t*c,int32_t*out,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  int32_t s0=a[i],s1=b[i],s2=c[i],d;
  __asm__ volatile("v_med3_i32 %0, %1, %2, %3\n\t":"=v"(d):"v"(s0),"v"(s1),"v"(s2));
  out[i]=d;
}
static int32_t med(int32_t a,int32_t b,int32_t c){int32_t lo=a<b?a:b,hi=a>b?a:b; if(c<lo)return lo; if(c>hi)return hi; return c;}
int main(void){
  int32_t*ha=(int32_t*)malloc(N*4),*hb=(int32_t*)malloc(N*4),*hc=(int32_t*)malloc(N*4),*ho=(int32_t*)malloc(N*4);
  for(int i=0;i<N;i++){ha[i]=(int32_t)(i*1103515245+12345);hb[i]=(int32_t)(-i*2654435761u+7);hc[i]=(int32_t)((i-128)*16777259);}
  int32_t*da,*db,*dc,*dd;
  GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&dd,N*4));
  GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
  kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,dd,N);
  GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
  GPU_CHECK(hipMemcpy(ho,dd,N*4,hipMemcpyDeviceToHost));
  int pass=1;for(int i=0;i<N;i++){int32_t e=med(ha[i],hb[i],hc[i]);if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,ho[i],e);pass=0;break;}}
  if(pass)printf("out[1]=%d\n",ho[1]); printf("%s\n",pass?"PASSED":"FAILED"); return pass?0:1;
}
