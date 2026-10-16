/*
 * HIP test for V_MUL_F32 (ENC_VOP2 single-precision multiply), with DPP and
 * SDWA variants.  gem5 (vop2.cc) routes this through vop2Helper, which applies
 * the DPP lane swizzle / SDWA operand selects before the op (D.f = S0 * S1).
 * All inputs are exactly-representable f32 (multiples of 0.5, small magnitude)
 * so every result is bit-exact.
 *
 *   base : v_mul_f32_e32 vd, va, vb              -> a[i] * b[i]
 *   dpp  : v_mul_f32_dpp ... quad_perm:[1,0,3,2] -> a[swizzled lane] * b[i]
 *   sdwa : v_mul_f32_sdwa ... DWORD selects      -> c[i] * b[i] (full-dword
 *          identity selects exercise the SDWA decode/exec path)
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_mul_f32 v_mul_f32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                          \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N=256, BLOCK_SIZE=64;

__global__ void kbase(const float*a,const float*b,float*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  float va=a[i],vb=b[i],vd;
  __asm__ volatile("v_mul_f32_e32 %0, %1, %2\n\t":"=v"(vd):"v"(va),"v"(vb));
  o[i]=vd;
}
__global__ void kdpp(const float*a,const float*b,float*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  float va=a[i],vb=b[i],vd;
  __asm__ volatile("v_mul_f32_dpp %0, %1, %2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(vd):"v"(va),"v"(vb));
  o[i]=vd;
}
__global__ void ksdwa(const float*c,const float*b,float*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  float vc=c[i],vb=b[i],vd;
  __asm__ volatile("v_mul_f32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD\n\t":"=v"(vd):"v"(vc),"v"(vb));
  o[i]=vd;
}
static uint32_t fbits(float f){uint32_t b;memcpy(&b,&f,4);return b;}
int main(void){
  float*a=(float*)malloc(N*4),*b=(float*)malloc(N*4),*c=(float*)malloc(N*4);
  float*ob=(float*)malloc(N*4),*od=(float*)malloc(N*4),*os=(float*)malloc(N*4);
  for(int i=0;i<N;i++){
    a[i]=(float)((i%17)-8)*0.5f;
    b[i]=(float)((i%9)-4)*0.5f;
    c[i]=(float)((i%7)-3)*0.5f;
  }
  float*da,*db,*dc,*dob,*dod,*dos;
  GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dc,N*4));
  GPU_CHECK(hipMalloc(&dob,N*4));GPU_CHECK(hipMalloc(&dod,N*4));GPU_CHECK(hipMalloc(&dos,N*4));
  GPU_CHECK(hipMemcpy(da,a,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,b,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,c,N*4,hipMemcpyHostToDevice));
  kbase<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dob,N); GPU_CHECK(hipGetLastError());
  kdpp<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dod,N); GPU_CHECK(hipGetLastError());
  ksdwa<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dc,db,dos,N); GPU_CHECK(hipGetLastError());
  GPU_CHECK(hipDeviceSynchronize());
  GPU_CHECK(hipMemcpy(ob,dob,N*4,hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(od,dod,N*4,hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(os,dos,N*4,hipMemcpyDeviceToHost));
  static const int perm[4]={1,0,3,2};
  int pass=1;
  for(int i=0;i<N;i++){
    uint32_t eb=fbits(a[i]*b[i]);
    if(fbits(ob[i])!=eb){fprintf(stderr,"FAIL base[%d] got %08x exp %08x\n",i,fbits(ob[i]),eb);pass=0;break;}
    int lane=i%BLOCK_SIZE,wbase=i-lane,sl=(lane&~3)+perm[lane&3];
    uint32_t ed=fbits(a[wbase+sl]*b[i]);
    if(fbits(od[i])!=ed){fprintf(stderr,"FAIL dpp[%d] got %08x exp %08x\n",i,fbits(od[i]),ed);pass=0;break;}
    uint32_t es=fbits(c[i]*b[i]);
    if(fbits(os[i])!=es){fprintf(stderr,"FAIL sdwa[%d] got %08x exp %08x\n",i,fbits(os[i]),es);pass=0;break;}
  }
  if(pass)printf("base[1]=%.3f dpp[1]=%.3f sdwa[1]=%.3f\n",ob[1],od[1],os[1]);
  printf("%s\n",pass?"PASSED":"FAILED");
  return pass?0:1;
}
