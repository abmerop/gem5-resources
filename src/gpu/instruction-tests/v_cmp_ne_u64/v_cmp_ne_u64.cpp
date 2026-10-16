/* V_CMP_NE_U64 (S0.u64!=S1.u64 mask). gem5: vop3_cmp.cc. _e64 writes per-lane mask. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const uint64_t*a,const uint64_t*b,int*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 uint64_t A_=a[i],B_=b[i]; uint64_t va=A_; uint64_t vb=B_; uint64_t mask;
 __asm__ volatile("v_cmp_ne_u64_e64 %0, %1, %2\n\t":"=s"(mask):"v"(va),"v"(vb));
 out[i]=(int)((mask>>threadIdx.x)&1ULL); }
int main(void){
 uint64_t*ha=(uint64_t*)malloc(N*sizeof(uint64_t)),*hb=(uint64_t*)malloc(N*sizeof(uint64_t));int*ho=(int*)malloc(N*sizeof(int));
 for(int i=0;i<N;i++){ha[i]=(uint64_t)((i%4==0)?(uint64_t)(i*0x1000001ull):(uint64_t)(i*2654435761ull));hb[i]=(uint64_t)((uint64_t)(i*0x1000001ull));}
 uint64_t*da,*db;int*dd;GPU_CHECK(hipMalloc(&da,N*sizeof(uint64_t)));GPU_CHECK(hipMalloc(&db,N*sizeof(uint64_t)));GPU_CHECK(hipMalloc(&dd,N*sizeof(int)));
 GPU_CHECK(hipMemcpy(da,ha,N*sizeof(uint64_t),hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*sizeof(uint64_t),hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dd,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,dd,N*sizeof(int),hipMemcpyDeviceToHost));
 int pass=1,nt=0;for(int i=0;i<N;i++){ uint64_t A=ha[i],B=hb[i];int e=(A!=B)?1:0;nt+=e;if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,ho[i],e);pass=0;break;}}
 if(pass)printf("true=%d/%d\n",nt,N);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
