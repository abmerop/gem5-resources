/* V_CVT_PK_U16_U32 (ENC_VOP3): D[15:0]=satu16(S0), D[31:16]=satu16(S1).
 * The conversion saturates (clamps) each u32 to [0, 0xFFFF]; it does NOT
 * truncate. gem5: vop3.cc. Pattern A (vector). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const uint32_t*a,const uint32_t*b,uint32_t*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
 uint32_t s0=a[i],s1=b[i],dv;
 __asm__ volatile("v_cvt_pk_u16_u32 %0, %1, %2\n\t":"=v"(dv):"v"(s0),"v"(s1));
 out[i]=dv;}
int main(void){
 uint32_t*ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
 for(int i=0;i<N;i++){ha[i]=(uint32_t)(i*2654435761u);hb[i]=(uint32_t)((128-i)*40503u+9u);}
 uint32_t*da,*db,*dd;
 GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dd,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dd,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,dd,N*4,hipMemcpyDeviceToHost));
 int pass=1;for(int i=0;i<N;i++){uint32_t lo=ha[i]>0xffffu?0xffffu:ha[i];uint32_t hi=hb[i]>0xffffu?0xffffu:hb[i];uint32_t e=(hi<<16)|lo;if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,ho[i],e);pass=0;break;}}
 if(pass)printf("out[1]=%08x\n",ho[1]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
