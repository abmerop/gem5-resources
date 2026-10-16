/*
 * HIP test for V_CVT_F32_I32 (VOP1 signed int -> f32), with DPP and SDWA.
 * gem5 (vop1.cc via vop1Helper): vdst = (float)src.i32.  vop1Helper applies
 * the DPP swizzle / SDWA src-select to src0.  Inputs have |x| < 2^24 so the
 * conversion is exact.
 *
 *   base : v_cvt_f32_i32_e32 vd, va
 *   dpp  : quad_perm:[1,0,3,2] swizzles src0 -> (float)a[swizzled]
 *   sdwa : src0_sel:DWORD (identity) -> (float)a[i]
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_f32_i32 v_cvt_f32_i32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
 "HIP error %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kbase(const int32_t*a,float*o,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int32_t s0=a[i];float vd;
 __asm__ volatile("v_cvt_f32_i32_e32 %0,%1\n\t":"=v"(vd):"v"(s0));o[i]=vd;}
__global__ void kdpp(const int32_t*a,float*o,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int32_t s0=a[i];float vd;
 __asm__ volatile("v_cvt_f32_i32_dpp %0,%1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(vd):"v"(s0));o[i]=vd;}
__global__ void ksdwa(const int32_t*a,float*o,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int32_t s0=a[i];float vd;
 __asm__ volatile("v_cvt_f32_i32_sdwa %0,%1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD\n\t":"=v"(vd):"v"(s0));o[i]=vd;}

int main(void){
 int32_t*ha=(int32_t*)malloc(N*4);float*hbase=(float*)malloc(N*4),*hdpp=(float*)malloc(N*4),*hsdwa=(float*)malloc(N*4);
 for(int i=0;i<N;i++) ha[i]=(int32_t)((i*131071)%(1<<23)) - (1<<22);
 int32_t*da;float*dbase,*ddpp,*dsdwa;
 GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&dbase,N*4));GPU_CHECK(hipMalloc(&ddpp,N*4));GPU_CHECK(hipMalloc(&dsdwa,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
 int g=N/BLOCK_SIZE;
 hipLaunchKernelGGL(kbase,dim3(g),dim3(BLOCK_SIZE),0,0,da,dbase,N);GPU_CHECK(hipGetLastError());
 hipLaunchKernelGGL(kdpp,dim3(g),dim3(BLOCK_SIZE),0,0,da,ddpp,N);GPU_CHECK(hipGetLastError());
 hipLaunchKernelGGL(ksdwa,dim3(g),dim3(BLOCK_SIZE),0,0,da,dsdwa,N);GPU_CHECK(hipGetLastError());
 GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(hbase,dbase,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hdpp,ddpp,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hsdwa,dsdwa,N*4,hipMemcpyDeviceToHost));
 static const int perm[4]={1,0,3,2};int pass=1;
 for(int i=0;i<N;i++){
  float eb=(float)ha[i]; if(hbase[i]!=eb){fprintf(stderr,"FAIL base[%d] %g exp %g\n",i,hbase[i],eb);pass=0;break;}
  int lane=i%BLOCK_SIZE,wb=i-lane,sl=(lane&~3)+perm[lane&3];
  float ed=(float)ha[wb+sl]; if(hdpp[i]!=ed){fprintf(stderr,"FAIL dpp[%d] %g exp %g\n",i,hdpp[i],ed);pass=0;break;}
  if(hsdwa[i]!=eb){fprintf(stderr,"FAIL sdwa[%d] %g exp %g\n",i,hsdwa[i],eb);pass=0;break;}}
 if(pass)printf("base[1]=%g dpp[1]=%g sdwa[1]=%g\n",hbase[1],hdpp[1],hsdwa[1]);
 printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
