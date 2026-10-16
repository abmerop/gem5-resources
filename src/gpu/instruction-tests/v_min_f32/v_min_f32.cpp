/*
 * HIP test for V_MIN_F32 (VOP2 f32 minimum), with DPP and SDWA.
 * gem5 (vop2.cc via vop2Helper): vdst = fmin(src0, src1).  vop2Helper applies
 * the DPP swizzle / SDWA src-select to SRC0.  Since min is commutative, the
 * dpp/sdwa kernels pass the same array for both operands so the result is
 * independent of which operand is swizzled.  Inputs are exact small floats.
 *
 *   base : v_min_f32_e32 vd, va, vb          min(a[i], b[i])
 *   dpp  : src0=src1=a, quad_perm:[1,0,3,2]  min(a[i], a[swizzled])
 *   sdwa : src0=src1=a, identity selects     min(a[i], a[i]) = a[i]
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_min_f32 v_min_f32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
 "HIP error %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kbase(const float*a,const float*b,float*o,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float s0=a[i],s1=b[i],vd;
 __asm__ volatile("v_min_f32_e32 %0,%1,%2\n\t":"=v"(vd):"v"(s0),"v"(s1));o[i]=vd;}
__global__ void kdpp(const float*a,float*o,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float s0=a[i],vd;
 __asm__ volatile("v_min_f32_dpp %0,%1,%1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(vd):"v"(s0));o[i]=vd;}
__global__ void ksdwa(const float*a,float*o,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float s0=a[i],vd;
 __asm__ volatile("v_min_f32_sdwa %0,%1,%1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD\n\t":"=v"(vd):"v"(s0));o[i]=vd;}

int main(void){
 float*ha=(float*)malloc(N*4),*hb=(float*)malloc(N*4),*hbase=(float*)malloc(N*4),*hdpp=(float*)malloc(N*4),*hsdwa=(float*)malloc(N*4);
 for(int i=0;i<N;i++){ha[i]=(float)((i%97)-48);hb[i]=(float)((i%61)-30)*2.0f;}
 float*da,*db,*dbase,*ddpp,*dsdwa;
 GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dbase,N*4));GPU_CHECK(hipMalloc(&ddpp,N*4));GPU_CHECK(hipMalloc(&dsdwa,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
 int g=N/BLOCK_SIZE;
 hipLaunchKernelGGL(kbase,dim3(g),dim3(BLOCK_SIZE),0,0,da,db,dbase,N);GPU_CHECK(hipGetLastError());
 hipLaunchKernelGGL(kdpp,dim3(g),dim3(BLOCK_SIZE),0,0,da,ddpp,N);GPU_CHECK(hipGetLastError());
 hipLaunchKernelGGL(ksdwa,dim3(g),dim3(BLOCK_SIZE),0,0,da,dsdwa,N);GPU_CHECK(hipGetLastError());
 GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(hbase,dbase,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hdpp,ddpp,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hsdwa,dsdwa,N*4,hipMemcpyDeviceToHost));
 static const int perm[4]={1,0,3,2};int pass=1;
 for(int i=0;i<N;i++){
  float eb=fminf(ha[i],hb[i]); if(hbase[i]!=eb){fprintf(stderr,"FAIL base[%d] %g exp %g\n",i,hbase[i],eb);pass=0;break;}
  int lane=i%BLOCK_SIZE,wb=i-lane,sl=(lane&~3)+perm[lane&3];
  float ed=fminf(ha[i],ha[wb+sl]); if(hdpp[i]!=ed){fprintf(stderr,"FAIL dpp[%d] %g exp %g\n",i,hdpp[i],ed);pass=0;break;}
  if(hsdwa[i]!=ha[i]){fprintf(stderr,"FAIL sdwa[%d] %g exp %g\n",i,hsdwa[i],ha[i]);pass=0;break;}}
 if(pass)printf("base[1]=%g dpp[1]=%g sdwa[1]=%g\n",hbase[1],hdpp[1],hsdwa[1]);
 printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;}
