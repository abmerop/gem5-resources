/* HIP test for V_MIN3_U32 (ENC_VOP3). D=min3(u32).
 * gem5: vop3.cc. Pattern A (vector): per-lane "v" operands. Exact check. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef uint32_t T;
#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const T*a,const T*b,const T*c,T*out,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  T s0=a[i],s1=b[i],s2=c[i],d;
  __asm__ volatile("v_min3_u32 %0, %1, %2, %3\n\t":"=v"(d):"v"(s0),"v"(s1),"v"(s2));
  out[i]=d;
}
static T ref(T a,T b,T c){ return (a<b?(a<c?a:c):(b<c?b:c)); }
int main(void){
  T*ha=(T*)malloc(N*sizeof(T)),*hb=(T*)malloc(N*sizeof(T)),*hc=(T*)malloc(N*sizeof(T)),*ho=(T*)malloc(N*sizeof(T));
  for(int i=0;i<N;i++){ha[i]=(T)(i*1103515245+12345);hb[i]=(T)((128-i)*2654435761u+7);hc[i]=(T)((i-64)*40503);}
  T*da,*db,*dc,*dd; GPU_CHECK(hipMalloc(&da,N*sizeof(T)));GPU_CHECK(hipMalloc(&db,N*sizeof(T)));
  GPU_CHECK(hipMalloc(&dc,N*sizeof(T)));GPU_CHECK(hipMalloc(&dd,N*sizeof(T)));
  GPU_CHECK(hipMemcpy(da,ha,N*sizeof(T),hipMemcpyHostToDevice));
  GPU_CHECK(hipMemcpy(db,hb,N*sizeof(T),hipMemcpyHostToDevice));
  GPU_CHECK(hipMemcpy(dc,hc,N*sizeof(T),hipMemcpyHostToDevice));
  kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,dd,N);
  GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
  GPU_CHECK(hipMemcpy(ho,dd,N*sizeof(T),hipMemcpyDeviceToHost));
  int pass=1; for(int i=0;i<N;i++){T e=ref(ha[i],hb[i],hc[i]); if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,ho[i],e);pass=0;break;}}
  if(pass)printf("out[1]=%u\n",ho[1]);
  printf("%s\n",pass?"PASSED":"FAILED");
  return pass?0:1;
}
