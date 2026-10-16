/*
 * HIP test for V_CVT_F64_U32 (ENC_VOP1, D.f64 = (f64)S0.u32), with DPP variant.
 * Pattern A.
 *
 *   base : v_cvt_f64_u32 vd[2], vsrc          -> (double)a[i]
 *   dpp  : v_cvt_f64_u32 vd[2], vsrc row_newbcast:0
 *          64-bit "DP ALU" DPP only permits row_newbcast on gfx950.  gem5
 *          (vega/insts/inst_util.hh dppInstImpl, SQ_DPP_ROW_NEWBCAST) maps
 *          row_newbcast:N so every lane of a 16-lane row reads src0 from lane
 *          (row*16 + N).  With N=0, every lane in a row converts the u32 from
 *          the row's first lane.
 *
 * Integer -> double conversions of the chosen inputs are exact, so results are
 * bit-exact.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_f64_u32 v_cvt_f64_u32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kbase(const uint32_t*a,double*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  uint32_t s0=a[i]; double d;
  __asm__ volatile("v_cvt_f64_u32 %0, %1\n\t":"=v"(d):"v"(s0));
  o[i]=d;
}
__global__ void kdpp(const uint32_t*a,double*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  uint32_t s0=a[i]; double d;
  __asm__ volatile("v_cvt_f64_u32 %0, %1 row_newbcast:0 row_mask:0xf bank_mask:0xf\n\t"
                   :"=v"(d):"v"(s0));
  o[i]=d;
}
int main(void){
  uint32_t*a=(uint32_t*)malloc(N*4);
  double*ob=(double*)malloc(N*8),*od=(double*)malloc(N*8);
  for(int i=0;i<N;i++) a[i]=(uint32_t)(i*7u+3u);
  uint32_t*da; double*dob,*dod;
  GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&dob,N*8));GPU_CHECK(hipMalloc(&dod,N*8));
  GPU_CHECK(hipMemcpy(da,a,N*4,hipMemcpyHostToDevice));
  kbase<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,dob,N); GPU_CHECK(hipGetLastError());
  kdpp<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,dod,N); GPU_CHECK(hipGetLastError());
  GPU_CHECK(hipDeviceSynchronize());
  GPU_CHECK(hipMemcpy(ob,dob,N*8,hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(od,dod,N*8,hipMemcpyDeviceToHost));
  int pass=1;
  for(int i=0;i<N;i++){
    double eb=(double)a[i];
    if(ob[i]!=eb){fprintf(stderr,"FAIL base[%d] got %f exp %f\n",i,ob[i],eb);pass=0;break;}
    int lane=i%BLOCK_SIZE,wbase=i-lane,row=lane/16,srcLane=wbase+row*16;
    double ed=(double)a[srcLane];
    if(od[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %f exp %f\n",i,od[i],ed);pass=0;break;}
  }
  if(pass)printf("base[1]=%.1f dpp[1]=%.1f\n",ob[1],od[1]);
  printf("%s\n",pass?"PASSED":"FAILED");
  return pass?0:1;
}
