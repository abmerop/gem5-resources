/* V_MED3_I16 (ENC_VOP3): signed 16-bit median-of-three. gem5: vop3.cc. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const int16_t*a,const int16_t*b,const int16_t*c,int16_t*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 int16_t s0=a[i],s1=b[i],s2=c[i],dv;
 __asm__ volatile("v_med3_i16 %0, %1, %2, %3\n\t":"=v"(dv):"v"(s0),"v"(s1),"v"(s2));
 out[i]=dv;}
static int16_t med(int16_t a,int16_t b,int16_t c){int16_t lo=a<b?a:b,hi=a>b?a:b;if(c<lo)return lo;if(c>hi)return hi;return c;}
int main(void){
 int16_t*ha=(int16_t*)malloc(N*2),*hb=(int16_t*)malloc(N*2),*hc=(int16_t*)malloc(N*2),*ho=(int16_t*)malloc(N*2);
 for(int i=0;i<N;i++){ha[i]=(int16_t)((i*277)-30000);hb[i]=(int16_t)(20000-(i*191));hc[i]=(int16_t)((i-128)*231);}
 int16_t*da,*db,*dc,*dd;GPU_CHECK(hipMalloc(&da,N*2));GPU_CHECK(hipMalloc(&db,N*2));GPU_CHECK(hipMalloc(&dc,N*2));GPU_CHECK(hipMalloc(&dd,N*2));
 GPU_CHECK(hipMemcpy(da,ha,N*2,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*2,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*2,hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,dd,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,dd,N*2,hipMemcpyDeviceToHost));
 int pass=1;for(int i=0;i<N;i++){int16_t e=med(ha[i],hb[i],hc[i]);if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,ho[i],e);pass=0;break;}}
 if(pass)printf("out[1]=%d\n",ho[1]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
