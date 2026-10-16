/*
 * HIP test for V_SUB_F16 (ENC_VOP2 half-precision), with DPP and SDWA variants.
 * gem5 (vop2.cc) routes this through vop2Helper, which applies the DPP lane
 * swizzle / SDWA operand selects before the op (D.f16 = S0 - S1).  All inputs are
 * exactly-representable f16 (multiples of 0.5, small magnitude) so every
 * result is bit-exact.
 *
 *   base : v_sub_f16_e32 vd, va, vb            -> op(a[i], b[i])
 *   dpp  : v_sub_f16_dpp ... quad_perm:[1,0,3,2] -> op(a[swizzled lane], b[i])
 *   sdwa : v_sub_f16_sdwa ... src0_sel:WORD_1   -> op(c[i] (high half), b[i])
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                          \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N=256, BLOCK_SIZE=64;

__global__ void kbase(const __half*a,const __half*b,__half*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  uint16_t ha,hb; __builtin_memcpy(&ha,a+i,2);__builtin_memcpy(&hb,b+i,2);
  uint32_t va=ha,vb=hb,vd;
  __asm__ volatile("v_sub_f16_e32 %0, %1, %2\n\t":"=v"(vd):"v"(va),"v"(vb));
  uint16_t hd=(uint16_t)(vd&0xffff); __builtin_memcpy(o+i,&hd,2);
}
__global__ void kdpp(const __half*a,const __half*b,__half*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  uint16_t ha,hb; __builtin_memcpy(&ha,a+i,2);__builtin_memcpy(&hb,b+i,2);
  uint32_t va=ha,vb=hb,vd;
  __asm__ volatile("v_sub_f16_dpp %0, %1, %2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(vd):"v"(va),"v"(vb));
  uint16_t hd=(uint16_t)(vd&0xffff); __builtin_memcpy(o+i,&hd,2);
}
__global__ void ksdwa(const __half*c,const __half*b,__half*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  uint16_t hc,hb; __builtin_memcpy(&hc,c+i,2);__builtin_memcpy(&hb,b+i,2);
  uint32_t packed=((uint32_t)hc)<<16, vb=hb, vd;
  __asm__ volatile("v_sub_f16_sdwa %0, %1, %2 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0\n\t":"=v"(vd):"v"(packed),"v"(vb));
  uint16_t hd=(uint16_t)(vd&0xffff); __builtin_memcpy(o+i,&hd,2);
}
static uint16_t hbits(__half h){uint16_t b;__builtin_memcpy(&b,&h,2);return b;}
static __half hop(float fx,float fy){ return __float2half(fx - fy); }
int main(void){
  __half*a=(__half*)malloc(N*2),*b=(__half*)malloc(N*2),*c=(__half*)malloc(N*2);
  __half*ob=(__half*)malloc(N*2),*od=(__half*)malloc(N*2),*os=(__half*)malloc(N*2);
  for(int i=0;i<N;i++){
    a[i]=__float2half((float)((i%17)-8)*0.5f);
    b[i]=__float2half((float)((i%9)-4)*0.5f);
    c[i]=__float2half((float)((i%7)-3)*0.5f);
  }
  __half*da,*db,*dc,*dob,*dod,*dos;
  GPU_CHECK(hipMalloc(&da,N*2));GPU_CHECK(hipMalloc(&db,N*2));GPU_CHECK(hipMalloc(&dc,N*2));
  GPU_CHECK(hipMalloc(&dob,N*2));GPU_CHECK(hipMalloc(&dod,N*2));GPU_CHECK(hipMalloc(&dos,N*2));
  GPU_CHECK(hipMemcpy(da,a,N*2,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,b,N*2,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,c,N*2,hipMemcpyHostToDevice));
  kbase<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dob,N); GPU_CHECK(hipGetLastError());
  kdpp<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dod,N); GPU_CHECK(hipGetLastError());
  ksdwa<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dc,db,dos,N); GPU_CHECK(hipGetLastError());
  GPU_CHECK(hipDeviceSynchronize());
  GPU_CHECK(hipMemcpy(ob,dob,N*2,hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(od,dod,N*2,hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(os,dos,N*2,hipMemcpyDeviceToHost));
  static const int perm[4]={1,0,3,2};
  int pass=1;
  for(int i=0;i<N;i++){
    float fa=__half2float(a[i]),fb=__half2float(b[i]),fc=__half2float(c[i]);
    uint16_t eb=hbits(hop(fa,fb));
    if(hbits(ob[i])!=eb){fprintf(stderr,"FAIL base[%d] got %04x exp %04x\n",i,hbits(ob[i]),eb);pass=0;break;}
    int lane=i%BLOCK_SIZE,wbase=i-lane,sl=(lane&~3)+perm[lane&3];
    float fasw=__half2float(a[wbase+sl]);
    uint16_t ed=hbits(hop(fasw,fb));
    if(hbits(od[i])!=ed){fprintf(stderr,"FAIL dpp[%d] got %04x exp %04x\n",i,hbits(od[i]),ed);pass=0;break;}
    uint16_t es=hbits(hop(fc,fb));
    if(hbits(os[i])!=es){fprintf(stderr,"FAIL sdwa[%d] got %04x exp %04x\n",i,hbits(os[i]),es);pass=0;break;}
  }
  if(pass)printf("base[1]=%.3f dpp[1]=%.3f sdwa[1]=%.3f\n",__half2float(ob[1]),__half2float(od[1]),__half2float(os[1]));
  printf("%s\n",pass?"PASSED":"FAILED");
  return pass?0:1;
}
