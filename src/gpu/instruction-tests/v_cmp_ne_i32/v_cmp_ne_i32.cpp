/* HIP test for V_CMP_NE_I32 (integer not-equal compare -> SGPR mask).
 * gem5: vop3_cmp.cc. _e64 form writes per-lane result to an SGPR mask. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const int32_t*a,const int32_t*b,int*out,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  int32_t va=a[i],vb=b[i]; uint64_t mask;
  __asm__ volatile("v_cmp_ne_i32_e64 %0, %1, %2\n\t":"=s"(mask):"v"(va),"v"(vb));
  out[i]=(int)((mask>>threadIdx.x)&1ULL);
}
int main(void){
  int32_t*ha=(int32_t*)malloc(N*4),*hb=(int32_t*)malloc(N*4); int*ho=(int*)malloc(N*sizeof(int));
  for(int i=0;i<N;i++){ha[i]=(int32_t)((i%9)-4); hb[i]=(int32_t)((i%5)-2);}
  int32_t*da,*db; int*dd;
  GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dd,N*sizeof(int)));
  GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
  kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dd,N);
  GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
  GPU_CHECK(hipMemcpy(ho,dd,N*sizeof(int),hipMemcpyDeviceToHost));
  int pass=1,nt=0;for(int i=0;i<N;i++){int e=(ha[i]!=hb[i])?1:0;nt+=e;if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,ho[i],e);pass=0;break;}}
  if(pass)printf("true=%d/%d\n",nt,N); printf("%s\n",pass?"PASSED":"FAILED"); return pass?0:1;
}
