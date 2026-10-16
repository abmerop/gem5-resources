/*
 * HIP test for V_CVT_I16_F16 (VOP1 f16->i16 convert) with DPP and SDWA.
 *
 * gem5 (vop1.cc): vdst[15:0] = int16_t(float(S0.f16))  (truncate toward zero)
 * via vop1Helper.  Inputs are small f16-exact integers so the result is the
 * same integer -> exact compare.
 *   base : v_cvt_i16_f16_e32 ; dpp quad_perm:[1,0,3,2] ; sdwa low-word identity
 * Variants exercised: base,dpp,sdwa
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void
kern(const uint16_t*a,uint16_t*ob,uint16_t*od,uint16_t*os,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint16_t x=a[i], db,dd,ds;
    __asm__ volatile("v_cvt_i16_f16_e32 %0, %1\n\t":"=v"(db):"v"(x));
    __asm__ volatile("v_cvt_i16_f16_dpp %0, %1 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t":"=v"(dd):"v"(x));
    __asm__ volatile("v_cvt_i16_f16_sdwa %0, %1 dst_sel:WORD_0 "
                     "dst_unused:UNUSED_PAD src0_sel:WORD_0\n\t":"=v"(ds):"v"(x));
    ob[i]=db; od[i]=dd; os[i]=ds;
}
static uint16_t hb(float v){ __half h=__float2half(v); uint16_t b; __builtin_memcpy(&b,&h,2); return b; }
int main(void){
    int *val=(int*)malloc(N*sizeof(int));
    uint16_t *ha=(uint16_t*)malloc(N*2),*hb_=(uint16_t*)malloc(N*2),
             *hd=(uint16_t*)malloc(N*2),*hs=(uint16_t*)malloc(N*2);
    for(int i=0;i<N;i++){ val[i]=(i*7)%201-100; ha[i]=hb((float)val[i]); }
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
        uint16_t eb=(uint16_t)(int16_t)val[i], ed=(uint16_t)(int16_t)val[base+sl], es=eb;
        if(hb_[i]!=eb){fprintf(stderr,"FAIL base[%d] got %04x exp %04x\n",i,hb_[i],eb);pass=0;}
        else if(hd[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %04x exp %04x\n",i,hd[i],ed);pass=0;}
        else if(hs[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got %04x exp %04x\n",i,hs[i],es);pass=0;}
    }
    if(pass)printf("base[1]=%04x dpp[1]=%04x sdwa[1]=%04x\n",hb_[1],hd[1],hs[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
