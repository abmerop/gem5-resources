/*
 * HIP test for V_SMFMAC_I32_32X32X64_I8 (sparse MFMA, i8 inputs, i32 acc,
 * gfx950).  Pattern D.
 *
 *   D.i32 = A_sparse.i8 * B.i8 + C.i32      (4:2 structured sparsity)
 *
 * Operand register layout (per wavefront, 64 lanes), from llvm-mc -mcpu=gfx950:
 *   v_smfmac_i32_32x32x64_i8 a[0:15], v[16:19], v[20:27], v28
 *     SRC0 (A, sparse): 4 VGPRs = 16 packed i8 per lane
 *     SRC1 (B, dense) : 8 VGPRs = 32 packed i8 per lane
 *     SRC2 (index)    : 1 VGPR  = 4:2 sparsity index
 *     C/D accumulator : 16 VGPRs = 16 i32 per lane (M*N/64 = 32*32/64)
 *
 * Mapping-independent verification: every A byte = 1 and every B byte = 1,
 * C = 0.  Each output accumulates exactly K/2 = 32 non-zero products (1*1),
 * independent of lane/register mapping and of the sparsity index selection.
 * Therefore every output element is exactly 32.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_smfmac_i32_32x32x64_i8 \
 *        v_smfmac_i32_32x32x64_i8.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

static const int GPRS_A=4, GPRS_B=8, GPRS_D=16;
typedef __attribute__((ext_vector_type(4)))  int int4_v;
typedef __attribute__((ext_vector_type(8)))  int int8_v;
typedef __attribute__((ext_vector_type(16))) int int16_v;

__global__ void
smfmac_kernel(const uint32_t*a_in,const uint32_t*b_in,const int32_t*c_in,
              int32_t*d_out,int num_waves){
    int wf=blockIdx.x,lane=threadIdx.x; if(wf>=num_waves)return;
    int ba=(wf*64+lane)*GPRS_A, bb=(wf*64+lane)*GPRS_B, bcd=(wf*64+lane)*GPRS_D;
    int4_v va; for(int i=0;i<4;i++) va[i]=(int)a_in[ba+i];
    int8_v vb; for(int i=0;i<8;i++) vb[i]=(int)b_in[bb+i];
    int16_v vc; for(int i=0;i<16;i++) vc[i]=c_in[bcd+i];
    uint32_t idx=0x44444444u;   /* index-independent (all B == 1) */
    __asm__ volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "v_smfmac_i32_32x32x64_i8 %0, %1, %2, %3\n\t"
        :"+v"(vc):"v"(va),"v"(vb),"v"(idx));
    for(int i=0;i<16;i++) d_out[bcd+i]=vc[i];
}

int main(void){
    const int num_waves=4,wf=64,threads=num_waves*wf;
    size_t ac=(size_t)threads*GPRS_A, bc=(size_t)threads*GPRS_B, cdc=(size_t)threads*GPRS_D;
    uint32_t*h_a=(uint32_t*)calloc(ac,4),*h_b=(uint32_t*)calloc(bc,4);
    int32_t*h_c=(int32_t*)calloc(cdc,4),*h_d=(int32_t*)calloc(cdc,4);
    for(int t=0;t<threads;t++){
        for(int r=0;r<GPRS_A;r++) h_a[t*GPRS_A+r]=0x01010101u;
        for(int r=0;r<GPRS_B;r++) h_b[t*GPRS_B+r]=0x01010101u;
    }
    uint32_t*d_a,*d_b; int32_t*d_c,*d_d;
    GPU_CHECK(hipMalloc(&d_a,ac*4));GPU_CHECK(hipMalloc(&d_b,bc*4));
    GPU_CHECK(hipMalloc(&d_c,cdc*4));GPU_CHECK(hipMalloc(&d_d,cdc*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,ac*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,bc*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,cdc*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d,0,cdc*4));
    hipLaunchKernelGGL(smfmac_kernel,dim3(num_waves),dim3(wf),0,0,d_a,d_b,d_c,d_d,num_waves);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_d,d_d,cdc*4,hipMemcpyDeviceToHost));
    const int32_t EXPECTED=32; int pass=1,checked=0;
    for(int t=0;t<threads&&pass;t++) for(int r=0;r<GPRS_D;r++){
        int32_t got=h_d[t*GPRS_D+r];
        if(got!=EXPECTED){fprintf(stderr,"FAIL t=%d vgpr=%d got %d exp %d\n",t,r,got,EXPECTED);pass=0;break;}
        checked++;
    }
    printf("v_smfmac_i32_32x32x64_i8: expected %d, checked %d\n",EXPECTED,checked);
    if(pass)printf("D[0]=%d D[last]=%d\n",h_d[0],h_d[cdc-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
