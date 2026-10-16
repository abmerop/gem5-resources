/*
 * HIP test for V_CMPX_O_F32 (VOPC/VOP3 compare-set-EXEC, f32).
 * gem5: per-lane EXEC/VCC bit = ordered (neither operand NaN).  EXEC saved/restored around the
 * _e64 instruction; the mask is captured to an SGPR and read per lane.
 * Variants: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr, "HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const float*a,const float*b,int*out,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  float va=a[i],vb=b[i]; uint64_t mask,saved;
  __asm__ volatile("s_mov_b64 %1, exec\n\t"
    "v_cmpx_o_f32_e64 %0, %2, %3\n\t"
    "s_mov_b64 exec, %1\n\t":"=s"(mask),"=s"(saved):"v"(va),"v"(vb));
  out[i]=(int)((mask>>threadIdx.x)&1ULL);
}
int main(void){
  float*h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4);int*h_o=(int*)malloc(N*4);
  for(int i=0;i<N;i++){
    float xa=(float)((i%9)-4), xb=(float)((i%7)-3);
    if(i%11==0) xa=NAN; if(i%13==0) xb=NAN;
    h_a[i]=xa; h_b[i]=xb;
  }
  float*d_a,*d_b;int*d_o;
  GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_o,N*4));
  GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
  GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
  kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_o,N);
  GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
  GPU_CHECK(hipMemcpy(h_o,d_o,N*4,hipMemcpyDeviceToHost));
  int pass=1;
  for(int i=0;i<N;i++){ float a=h_a[i],b=h_b[i]; (void)a;(void)b;
    int exp=((!isnan(a) && !isnan(b)))?1:0;
    if(h_o[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_o[i],exp);pass=0;break;}
  }
  if(pass)printf("out[0]=%d out[%d]=%d\n",h_o[0],N-1,h_o[N-1]);
  printf("%s\n",pass?"PASSED":"FAILED");
  return pass?0:1;
}
