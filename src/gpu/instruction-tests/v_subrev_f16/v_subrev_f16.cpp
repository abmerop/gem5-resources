/*
 * HIP test for V_SUBREV_F16 (VOP2 reversed f16 subtract), with DPP and SDWA.
 *
 * gem5 (vop2.cc via vop2Helper): vdst = fplibSub(S1, S0) = src1 - src0 (f16).
 * The generic vop2Helper applies the DPP swizzle / SDWA src-select to SRC0.
 * Inputs are small exactly-representable f16 values so the result is exact.
 *
 *   base : v_subrev_f16_e32 vd, va, vb        vd = b[i] - a[i]
 *   dpp  : quad_perm:[1,0,3,2] swizzles src0=a -> vd = b[i] - a[swizzled]
 *   sdwa : src0_sel/src1_sel:WORD_0 (identity) -> vd = b[i] - a[i]
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_subrev_f16 v_subrev_f16.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kbase(const _Float16 *a, const _Float16 *b, uint32_t *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    _Float16 s0=a[i],s1=b[i]; uint32_t vd;
    __asm__ volatile("v_subrev_f16_e32 %0, %1, %2\n\t":"=v"(vd):"v"(s0),"v"(s1));
    out[i]=vd;
}
__global__ void
kdpp(const _Float16 *a, const _Float16 *b, uint32_t *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    _Float16 s0=a[i],s1=b[i]; uint32_t vd;
    __asm__ volatile("v_subrev_f16_dpp %0, %1, %2 quad_perm:[1,0,3,2] "
        "row_mask:0xf bank_mask:0xf\n\t":"=v"(vd):"v"(s0),"v"(s1));
    out[i]=vd;
}
__global__ void
ksdwa(const _Float16 *a, const _Float16 *b, uint32_t *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    _Float16 s0=a[i],s1=b[i]; uint32_t vd;
    __asm__ volatile("v_subrev_f16_sdwa %0, %1, %2 dst_sel:DWORD "
        "dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0\n\t"
        :"=v"(vd):"v"(s0),"v"(s1));
    out[i]=vd;
}

static uint16_t sub_bits(_Float16 b, _Float16 a){
    _Float16 r=(_Float16)((float)b-(float)a); uint16_t bits; memcpy(&bits,&r,2); return bits;
}

int main(void){
    _Float16 *h_a=(_Float16*)malloc(N*2),*h_b=(_Float16*)malloc(N*2);
    uint32_t *h_base=(uint32_t*)malloc(N*4),*h_dpp=(uint32_t*)malloc(N*4),*h_sdwa=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ h_a[i]=(_Float16)((float)((i%13)-6)); h_b[i]=(_Float16)((float)((i%9)-4)); }
    _Float16 *d_a,*d_b; uint32_t *d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));
    GPU_CHECK(hipMalloc(&d_base,N*4));GPU_CHECK(hipMalloc(&d_dpp,N*4));GPU_CHECK(hipMalloc(&d_sdwa,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    int g=N/BLOCK_SIZE;
    hipLaunchKernelGGL(kbase,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_base,N);GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kdpp,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_dpp,N);GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(ksdwa,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_sdwa,N);GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base,d_base,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp,d_dpp,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa,d_sdwa,N*4,hipMemcpyDeviceToHost));
    static const int perm[4]={1,0,3,2}; int pass=1;
    for(int i=0;i<N;i++){
        uint16_t eb=sub_bits(h_b[i],h_a[i]);
        if((uint16_t)h_base[i]!=eb){fprintf(stderr,"FAIL base[%d] got 0x%04x exp 0x%04x\n",i,(uint16_t)h_base[i],eb);pass=0;break;}
        int lane=i%BLOCK_SIZE,wb=i-lane,sl=(lane&~3)+perm[lane&3];
        uint16_t ed=sub_bits(h_b[i],h_a[wb+sl]);
        if((uint16_t)h_dpp[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got 0x%04x exp 0x%04x\n",i,(uint16_t)h_dpp[i],ed);pass=0;break;}
        uint16_t es=sub_bits(h_b[i],h_a[i]);
        if((uint16_t)h_sdwa[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got 0x%04x exp 0x%04x\n",i,(uint16_t)h_sdwa[i],es);pass=0;break;}
    }
    if(pass)printf("base[1]=0x%04x dpp[1]=0x%04x sdwa[1]=0x%04x\n",(uint16_t)h_base[1],(uint16_t)h_dpp[1],(uint16_t)h_sdwa[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
