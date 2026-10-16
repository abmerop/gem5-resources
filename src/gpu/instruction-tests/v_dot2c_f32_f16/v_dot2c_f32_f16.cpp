/*
 * HIP test for V_DOT2C_F32_F16 (ENC_VOP2 dot-product-accumulate), with DPP
 * variant.  gem5 (vop2.cc) computes, per lane (vdst is read-write):
 *
 *   vdst.f32 += f16_to_f32(S0[15:0])  * f16_to_f32(S1[15:0]);
 *   vdst.f32 += f16_to_f32(S0[31:16]) * f16_to_f32(S1[31:16]);
 *
 * (two sequential accumulations).  SDWA is explicitly unsupported by gem5 for
 * this opcode, so only base + DPP variants are exercised.
 *
 *   base : v_dot2c_f32_f16_e32 vd, va, vb       -> acc + a.lo*b.lo + a.hi*b.hi
 *   dpp  : v_dot2c_f32_f16_dpp ... quad_perm    -> a from swizzled lane
 *
 * Inputs are exactly-representable f16 (multiples of 0.5, small) so every f32
 * result is bit-exact.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_dot2c_f32_f16 v_dot2c_f32_f16.cpp
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                          \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N=256, BLOCK_SIZE=64;

__global__ void kbase(const uint32_t*a,const uint32_t*b,const float*acc,float*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  uint32_t va=a[i],vb=b[i]; float vd=acc[i];
  __asm__ volatile("v_dot2c_f32_f16_e32 %0, %1, %2\n\t":"+v"(vd):"v"(va),"v"(vb));
  o[i]=vd;
}
__global__ void kdpp(const uint32_t*a,const uint32_t*b,const float*acc,float*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  uint32_t va=a[i],vb=b[i]; float vd=acc[i];
  __asm__ volatile("v_dot2c_f32_f16_dpp %0, %1, %2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"+v"(vd):"v"(va),"v"(vb));
  o[i]=vd;
}
static uint32_t fbits(float f){uint32_t b;memcpy(&b,&f,4);return b;}
static uint32_t pack2(float lo,float hi){
  __half hl=__float2half(lo),hh=__float2half(hi); uint16_t bl,bh;
  memcpy(&bl,&hl,2);memcpy(&bh,&hh,2); return ((uint32_t)bh<<16)|bl;
}
static float dot(uint32_t a,uint32_t b,float acc){
  uint16_t a0=a&0xffff,a1=a>>16,b0=b&0xffff,b1=b>>16; __half ha0,ha1,hb0,hb1;
  memcpy(&ha0,&a0,2);memcpy(&ha1,&a1,2);memcpy(&hb0,&b0,2);memcpy(&hb1,&b1,2);
  float r=acc; r+=__half2float(ha0)*__half2float(hb0); r+=__half2float(ha1)*__half2float(hb1);
  return r;
}
int main(void){
  uint32_t*a=(uint32_t*)malloc(N*4),*b=(uint32_t*)malloc(N*4);
  float*acc=(float*)malloc(N*4),*ob=(float*)malloc(N*4),*od=(float*)malloc(N*4);
  for(int i=0;i<N;i++){
    a[i]=pack2((float)((i%9)-4)*0.5f,(float)((i%7)-3)*0.5f);
    b[i]=pack2((float)((i%5)-2)*0.5f,(float)((i%11)-5)*0.5f);
    acc[i]=(float)((i%13)-6)*0.5f;
  }
  uint32_t*da,*db; float*dacc,*dob,*dod;
  GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dacc,N*4));
  GPU_CHECK(hipMalloc(&dob,N*4));GPU_CHECK(hipMalloc(&dod,N*4));
  GPU_CHECK(hipMemcpy(da,a,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,b,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dacc,acc,N*4,hipMemcpyHostToDevice));
  kbase<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dacc,dob,N); GPU_CHECK(hipGetLastError());
  kdpp<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dacc,dod,N); GPU_CHECK(hipGetLastError());
  GPU_CHECK(hipDeviceSynchronize());
  GPU_CHECK(hipMemcpy(ob,dob,N*4,hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(od,dod,N*4,hipMemcpyDeviceToHost));
  static const int perm[4]={1,0,3,2};
  int pass=1;
  for(int i=0;i<N;i++){
    uint32_t eb=fbits(dot(a[i],b[i],acc[i]));
    if(fbits(ob[i])!=eb){fprintf(stderr,"FAIL base[%d] got %08x exp %08x\n",i,fbits(ob[i]),eb);pass=0;break;}
    int lane=i%BLOCK_SIZE,wbase=i-lane,sl=(lane&~3)+perm[lane&3];
    uint32_t ed=fbits(dot(a[wbase+sl],b[i],acc[i]));
    if(fbits(od[i])!=ed){fprintf(stderr,"FAIL dpp[%d] got %08x exp %08x\n",i,fbits(od[i]),ed);pass=0;break;}
  }
  if(pass)printf("base[1]=%.3f dpp[1]=%.3f\n",ob[1],od[1]);
  printf("%s\n",pass?"PASSED":"FAILED");
  return pass?0:1;
}
