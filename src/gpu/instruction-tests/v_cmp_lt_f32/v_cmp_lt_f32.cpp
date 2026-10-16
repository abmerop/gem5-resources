/* V_CMP_LT_F32 (S0.f32<S1.f32 mask). gem5: vop3_cmp.cc. _e64 writes per-lane mask. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const float*a,const float*b,int*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 float A_=a[i],B_=b[i]; float va=A_; float vb=B_;
 uint64_t mask;
 __asm__ volatile(
        "v_cmp_lt_f32_e64 %0, %1, %2\n\t"
  : "=s"(mask) : "v"(va),"v"(vb));
 out[i]=(int)((mask>>threadIdx.x)&1ULL); }
int main(void){
 float*ha=(float*)malloc(N*sizeof(float)),*hb=(float*)malloc(N*sizeof(float));int*ho=(int*)malloc(N*sizeof(int));
 for(int i=0;i<N;i++){ha[i]=(float)((float)((i%11)-5)*0.25f);hb[i]=(float)((float)((i%9)-4)*0.5f);}
 float*da,*db;int*dd;GPU_CHECK(hipMalloc(&da,N*sizeof(float)));GPU_CHECK(hipMalloc(&db,N*sizeof(float)));GPU_CHECK(hipMalloc(&dd,N*sizeof(int)));
 GPU_CHECK(hipMemcpy(da,ha,N*sizeof(float),hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*sizeof(float),hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dd,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,dd,N*sizeof(int),hipMemcpyDeviceToHost));
 int pass=1,nt=0;for(int i=0;i<N;i++){ float A=ha[i],B=hb[i];int e=(A < B)?1:0;nt+=e;if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,ho[i],e);pass=0;break;}}
 if(pass)printf("true=%d/%d\n",nt,N);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
