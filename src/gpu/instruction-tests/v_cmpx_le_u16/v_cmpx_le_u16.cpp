/* HIP test for V_CMPX_LE_U16 (VOPC compare-set-EXEC; EXEC saved/restored). Variants: base. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP error :0 \n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const uint32_t*a,const uint32_t*b,int*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; uint32_t va=a[i],vb=b[i]; uint64_t m,sv;
 __asm__ volatile("s_mov_b64 %1, exec\n\tv_cmpx_le_u16_e64 %0, %2, %3\n\ts_mov_b64 exec, %1\n\t"
  :"=s"(m),"=s"(sv):"v"(va),"v"(vb)); out[i]=(int)((m>>threadIdx.x)&1ULL);}
int main(void){uint32_t*ha=(uint32_t*)malloc(N*sizeof(uint32_t)),*hb=(uint32_t*)malloc(N*sizeof(uint32_t));int*ho=(int*)malloc(N*4);
 for(int i=0;i<N;i++){ha[i]=(uint32_t)(uint16_t)(i%7);hb[i]=(uint32_t)(uint16_t)(i%5);}
 uint32_t*da,*db;int*do_;GPU_CHECK(hipMalloc(&da,N*sizeof(uint32_t)));GPU_CHECK(hipMalloc(&db,N*sizeof(uint32_t)));GPU_CHECK(hipMalloc(&do_,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*sizeof(uint32_t),hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*sizeof(uint32_t),hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,do_,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,do_,N*4,hipMemcpyDeviceToHost));int pass=1;
 for(int i=0;i<N;i++){uint32_t a=ha[i],b=hb[i];(void)a;(void)b;int exp=((uint16_t)a <= (uint16_t)b)?1:0;
  if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,ho[i],exp);pass=0;break;}}
 if(pass)printf("out[0]=%d out[%d]=%d\n",ho[0],N-1,ho[N-1]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
