/*
 * HIP test for V_SUB_U32 (ENC_VOP2, D.u = S0.u - S1.u), with DPP and SDWA
 * variants.  Pattern A.
 *
 * gem5 (common/insts/vop2.cc Inst_VOP2__V_SUB_U32) routes through vop2Helper,
 * which applies the DPP lane swizzle / SDWA operand selects before the op.
 *
 *   base : v_sub_u32 vd, va, vb                    -> a[i] - b[i]  (u32 wrap)
 *   dpp  : v_sub_u32 ... quad_perm:[1,0,3,2]       -> a[swizzled lane] - b[i]
 *   sdwa : v_sub_u32 ... src0_sel:WORD_1           -> (a_hi16) - b[i]
 *
 * All arithmetic is unsigned 32-bit with modular wraparound, so results are
 * bit-exact.  SDWA selects are zero-extended by gem5 (SEXT=0).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_sub_u32 v_sub_u32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                          \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N=256, BLOCK_SIZE=64;

__global__ void kbase(const uint32_t*a,const uint32_t*b,uint32_t*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  uint32_t va=a[i],vb=b[i],vd;
  __asm__ volatile("v_sub_u32 %0, %1, %2\n\t":"=v"(vd):"v"(va),"v"(vb));
  o[i]=vd;
}
__global__ void kdpp(const uint32_t*a,const uint32_t*b,uint32_t*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  uint32_t va=a[i],vb=b[i],vd;
  __asm__ volatile("v_sub_u32 %0, %1, %2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t"
                   :"=v"(vd):"v"(va),"v"(vb));
  o[i]=vd;
}
__global__ void ksdwa(const uint32_t*a,const uint32_t*b,uint32_t*o,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
  /* pack the test value into the high word so src0_sel:WORD_1 extracts it */
  uint32_t packed=(a[i]&0xFFFFu)<<16, vb=b[i], vd;
  __asm__ volatile("v_sub_u32 %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:WORD_1 src1_sel:DWORD\n\t"
                   :"=v"(vd):"v"(packed),"v"(vb));
  o[i]=vd;
}

int main(void){
  uint32_t*a=(uint32_t*)malloc(N*4),*b=(uint32_t*)malloc(N*4);
  uint32_t*ob=(uint32_t*)malloc(N*4),*od=(uint32_t*)malloc(N*4),*os=(uint32_t*)malloc(N*4);
  for(int i=0;i<N;i++){
    a[i]=(uint32_t)(i*2654435761u+12345u);
    b[i]=(uint32_t)((N-i)*40503u+7u);
  }
  uint32_t*da,*db,*dob,*dod,*dos;
  GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));
  GPU_CHECK(hipMalloc(&dob,N*4));GPU_CHECK(hipMalloc(&dod,N*4));GPU_CHECK(hipMalloc(&dos,N*4));
  GPU_CHECK(hipMemcpy(da,a,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,b,N*4,hipMemcpyHostToDevice));
  kbase<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dob,N); GPU_CHECK(hipGetLastError());
  kdpp<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dod,N); GPU_CHECK(hipGetLastError());
  ksdwa<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dos,N); GPU_CHECK(hipGetLastError());
  GPU_CHECK(hipDeviceSynchronize());
  GPU_CHECK(hipMemcpy(ob,dob,N*4,hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(od,dod,N*4,hipMemcpyDeviceToHost));
  GPU_CHECK(hipMemcpy(os,dos,N*4,hipMemcpyDeviceToHost));
  static const int perm[4]={1,0,3,2};
  int pass=1;
  for(int i=0;i<N;i++){
    uint32_t eb=a[i]-b[i];
    if(ob[i]!=eb){fprintf(stderr,"FAIL base[%d] got %08x exp %08x\n",i,ob[i],eb);pass=0;break;}
    int lane=i%BLOCK_SIZE,wbase=i-lane,sl=(lane&~3)+perm[lane&3];
    uint32_t ed=a[wbase+sl]-b[i];
    if(od[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %08x exp %08x\n",i,od[i],ed);pass=0;break;}
    uint32_t es=(a[i]&0xFFFFu)-b[i];
    if(os[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got %08x exp %08x\n",i,os[i],es);pass=0;break;}
  }
  if(pass)printf("base[1]=%08x dpp[1]=%08x sdwa[1]=%08x\n",ob[1],od[1],os[1]);
  printf("%s\n",pass?"PASSED":"FAILED");
  return pass?0:1;
}
