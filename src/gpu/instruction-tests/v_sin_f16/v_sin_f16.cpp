/*
 * HIP test for V_SIN_F16 (VOP1 f16 sine) with DPP and SDWA variants.
 *
 * gem5 (vop1.cc, via vop1Helper) computes per lane:
 *     float vf32 = float(mxfloat16(src));
 *     float vd32 = std::sin(vf32 * 2.0f * (float)M_PI);   // input in revs
 *     vdst = f16bits(mxfloat16(vd32));
 * vop1Helper applies the DPP lane swizzle / SDWA source-select to src0 before
 * the sine, exactly like V_RCP_F16.  The host reference replicates the float
 * math; comparison is done on the decoded f16 values with a small absolute
 * tolerance (f16 rounding of a transcendental result).
 *   base : v_sin_f16_e32
 *   dpp  : quad_perm:[1,0,3,2]
 *   sdwa : dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:WORD_0
 * Variants exercised: base,dpp,sdwa
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void
kern(const uint16_t*a,uint16_t*ob,uint16_t*od,uint16_t*os,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint16_t x=a[i], db,dd,ds;
    __asm__ volatile("v_sin_f16_e32 %0, %1\n\t":"=v"(db):"v"(x));
    __asm__ volatile("v_sin_f16_dpp %0, %1 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t":"=v"(dd):"v"(x));
    __asm__ volatile("v_sin_f16_sdwa %0, %1 dst_sel:WORD_0 "
                     "dst_unused:UNUSED_PAD src0_sel:WORD_0\n\t":"=v"(ds):"v"(x));
    ob[i]=db; od[i]=dd; os[i]=ds;
}
static uint16_t hbits(float v){ __half h=__float2half(v); uint16_t b; __builtin_memcpy(&b,&h,2); return b; }
static float hval(uint16_t b){ __half h; __builtin_memcpy(&h,&b,2); return __half2float(h); }
static float sinref(uint16_t xb){ float vf=hval(xb); return sinf(vf*2.0f*(float)M_PI); }
static int close16(uint16_t g,float e){ float gf=hval(g); return fabsf(gf-e)<=0.01f; }
int main(void){
    uint16_t *ha=(uint16_t*)malloc(N*2),*hb_=(uint16_t*)malloc(N*2),
             *hd=(uint16_t*)malloc(N*2),*hs=(uint16_t*)malloc(N*2);
    for(int i=0;i<N;i++) ha[i]=hbits(((float)(i%64)-32.0f)*0.0625f); /* revolutions */
    uint16_t *da,*db,*dd,*ds;
    GPU_CHECK(hipMalloc(&da,N*2));GPU_CHECK(hipMalloc(&db,N*2));
    GPU_CHECK(hipMalloc(&dd,N*2));GPU_CHECK(hipMalloc(&ds,N*2));
    GPU_CHECK(hipMemcpy(da,ha,N*2,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dd,ds,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(hb_,db,N*2,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hd,dd,N*2,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hs,ds,N*2,hipMemcpyDeviceToHost));
    const int perm[4]={1,0,3,2}; int pass=1;
    for(int i=0;i<N&&pass;i++){
        int lane=i%BLOCK_SIZE, base=i-lane, sl=(lane&~3)|perm[lane&3];
        float eb=sinref(ha[i]), ed=sinref(ha[base+sl]), es=eb;
        if(!close16(hb_[i],eb)){fprintf(stderr,"FAIL base[%d] got %g exp %g\n",i,hval(hb_[i]),eb);pass=0;}
        else if(!close16(hd[i],ed)){fprintf(stderr,"FAIL dpp[%d] got %g exp %g\n",i,hval(hd[i]),ed);pass=0;}
        else if(!close16(hs[i],es)){fprintf(stderr,"FAIL sdwa[%d] got %g exp %g\n",i,hval(hs[i]),es);pass=0;}
    }
    if(pass)printf("base[3]=%g dpp[3]=%g sdwa[3]=%g\n",hval(hb_[3]),hval(hd[3]),hval(hs[3]));
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
