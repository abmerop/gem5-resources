/*
 * HIP test for V_MAD_MIXHI_F16 (ENC_VOP3P, mixed-precision fused multiply-add,
 * f16 result written to the HIGH 16 bits of the destination).  Pattern A.
 *
 * Per gem5 src/arch/amdgpu/common/insts/vop3p.cc, V_MAD_MIXHI_F16 computes:
 *
 *   tmp.f32    = fma(SRC0, SRC1, SRC2)   (op_sel_hi=0 -> all sources are f32)
 *   VDST[31:16]= f32_to_f16(tmp)         (narrow, write to HIGH half)
 *   VDST[15:0] = unchanged               (low half preserved -- vdst.read())
 *
 * On gfx950 the LLVM assembler spells this v_fma_mixhi_f16; gem5 decodes that
 * encoding as Inst_VOP3P__V_MAD_MIXHI_F16.  Inline asm is used to force the
 * MIXHI form deterministically with all-f32 sources.  All operands are strictly
 * positive so the exact-zero-sum -0.0 quirk never fires; small integers keep the
 * f16 narrowing exact.  A "+v" sentinel proves the low half is preserved.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_mad_mixhi_f16 v_mad_mixhi_f16.cpp
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define GPU_CHECK(cmd) do{hipError_t e=(cmd); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);}}while(0)

static const int N=256, BLOCK_SIZE=64;
static const uint32_t LOW_SENTINEL=0xB33Fu;   /* must survive in bits[15:0] */

__global__ void
kern(const float*a,const float*b,const float*c,uint32_t*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float s0=a[i], s1=b[i], s2=c[i];
    uint32_t vd=LOW_SENTINEL;   /* low half sentinel; high half will be written */
    __asm__ volatile("v_fma_mixhi_f16 %0,%1,%2,%3\n\t"
                     :"+v"(vd):"v"(s0),"v"(s1),"v"(s2));
    o[i]=vd;
}

int main(void){
    float*ha=(float*)malloc(N*4),*hb=(float*)malloc(N*4),*hc=(float*)malloc(N*4);
    uint32_t*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ ha[i]=(float)((i%5)+1); hb[i]=(float)((i%4)+1); hc[i]=(float)((i%7)+1); }
    float*da,*db,*dc; uint32_t*do_;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&do_,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,do_,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,do_,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        __half r=__float2half(fmaf(ha[i],hb[i],hc[i]));
        uint16_t hb16; memcpy(&hb16,&r,2);
        uint32_t exp=((uint32_t)hb16<<16)|LOW_SENTINEL;
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,ho[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",ho[0],N-1,ho[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
