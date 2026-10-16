/* V_CMP_LE_I32 (S0.i32<=S1.i32 mask). gem5: vop3_cmp.cc. _e64 writes per-lane mask. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const int32_t*a,const int32_t*b,int*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 int32_t A_=a[i],B_=b[i]; int32_t va=A_; int32_t vb=B_;
 uint64_t mask;
 __asm__ volatile(
        "v_cmp_le_i32_e64 %0, %1, %2\n\t"
  : "=s"(mask) : "v"(va),"v"(vb));
 out[i]=(int)((mask>>threadIdx.x)&1ULL); }
int main(void){
 int32_t*ha=(int32_t*)malloc(N*sizeof(int32_t)),*hb=(int32_t*)malloc(N*sizeof(int32_t));int*ho=(int*)malloc(N*sizeof(int));
 for(int i=0;i<N;i++){ha[i]=(int32_t)((int32_t)((i%17)-8));hb[i]=(int32_t)((int32_t)((i%11)-5));}
 int32_t*da,*db;int*dd;GPU_CHECK(hipMalloc(&da,N*sizeof(int32_t)));GPU_CHECK(hipMalloc(&db,N*sizeof(int32_t)));GPU_CHECK(hipMalloc(&dd,N*sizeof(int)));
 GPU_CHECK(hipMemcpy(da,ha,N*sizeof(int32_t),hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*sizeof(int32_t),hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dd,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,dd,N*sizeof(int),hipMemcpyDeviceToHost));
 int pass=1,nt=0;for(int i=0;i<N;i++){ int32_t A=ha[i],B=hb[i];int e=(A <= B)?1:0;nt+=e;if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,ho[i],e);pass=0;break;}}
 if(pass)printf("true=%d/%d\n",nt,N);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
