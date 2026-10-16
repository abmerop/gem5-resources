/* HIP test for V_CMPX_NE_I32 (compare-set-EXEC; EXEC saved/restored). Variants: base. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP error :0 \n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const int32_t*a,const int32_t*b,int*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int32_t va=a[i],vb=b[i];uint64_t m,sv;
 __asm__ volatile("s_mov_b64 %1, exec\n\tv_cmpx_ne_i32_e64 %0, %2, %3\n\ts_mov_b64 exec, %1\n\t"
  :"=s"(m),"=s"(sv):"v"(va),"v"(vb)); out[i]=(int)((m>>threadIdx.x)&1ULL);}
int main(void){int32_t*ha=(int32_t*)malloc(N*4),*hb=(int32_t*)malloc(N*4);int*ho=(int*)malloc(N*4);
 for(int i=0;i<N;i++){ha[i]=(int32_t)(i*131-500);hb[i]=(i%2)?ha[i]:(int32_t)(i*131-501);}
 int32_t*da,*db;int*do_;GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&do_,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,do_,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,do_,N*4,hipMemcpyDeviceToHost));int pass=1;
 for(int i=0;i<N;i++){int exp=(ha[i]!=hb[i])?1:0; if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,ho[i],exp);pass=0;break;}}
 if(pass)printf("out[0]=%d out[%d]=%d\n",ho[0],N-1,ho[N-1]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
