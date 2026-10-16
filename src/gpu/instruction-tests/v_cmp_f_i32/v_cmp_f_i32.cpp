/* V_CMP_F_I32 (VOPC compare, mask = 0 unconditionally -- the "false" compare).
 * gem5: vop3_cmp.cc. _e64 writes per-lane SGPR mask; every lane reads 0. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const int*a,const int*b,int*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 int va=a[i],vb=b[i];
 uint64_t mask;
 __asm__ volatile(
        "v_cmp_f_i32_e64 %0, %1, %2\n\t"
  : "=s"(mask) : "v"(va),"v"(vb));
 out[i]=(int)((mask>>threadIdx.x)&1ULL); }
int main(void){
 int*ha=(int*)malloc(N*sizeof(int)),*hb=(int*)malloc(N*sizeof(int));int*ho=(int*)malloc(N*sizeof(int));
 for(int i=0;i<N;i++){ha[i]=i-128;hb[i]=(i*7)%53-26;}
 int*da,*db,*dd;GPU_CHECK(hipMalloc(&da,N*sizeof(int)));GPU_CHECK(hipMalloc(&db,N*sizeof(int)));GPU_CHECK(hipMalloc(&dd,N*sizeof(int)));
 GPU_CHECK(hipMemcpy(da,ha,N*sizeof(int),hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*sizeof(int),hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dd,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,dd,N*sizeof(int),hipMemcpyDeviceToHost));
 int pass=1;for(int i=0;i<N;i++){int e=0;if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,ho[i],e);pass=0;break;}}
 if(pass)printf("all-false ok (out[0]=%d)\n",ho[0]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
