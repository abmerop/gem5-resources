/*
 * HIP test for v_smfmac_f32_32x32x64_bf8_fp8 (sparse MFMA, gfx950/CDNA4).
 * Uniform-input trick: A=1.0, B=1.0, C=0, index-independent ->
 * every output = K/2 = 32 (4:2 sparsity). acc in AGPR ("+a").
 * Widths (llvm-mc gfx950): A 4 VGPR, B 8 VGPR, C/D 16 AGPR, idx 1 VGPR.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);}} while(0)
static const int GA=4, GB=8, CD=16;
typedef __attribute__((ext_vector_type(4))) int int4_v;
typedef __attribute__((ext_vector_type(8))) int int8_v;
typedef __attribute__((ext_vector_type(16))) float float16_v;
__global__ void smfmac_kernel(const uint32_t*a_in,const uint32_t*b_in,
        const float*c_in,float*d_out,int num_waves) {
    int wf_id=blockIdx.x, lane=threadIdx.x;
    if(wf_id>=num_waves) return;
    int base=wf_id*64+lane;
    int4_v va;
    for (int i=0;i<4;i++) va[i]=(int)a_in[base*4+i];
    int8_v vb;
    for (int i=0;i<8;i++) vb[i]=(int)b_in[base*8+i];
    float16_v vc;
    for (int i=0;i<16;i++) vc[i]=(float)c_in[base*16+i];
    uint32_t idx=0x44444444u;
    __asm__ volatile("s_waitcnt vmcnt(0)\n\t"
        "v_smfmac_f32_32x32x64_bf8_fp8 %0, %1, %2, %3\n\t" : "+a"(vc) : "v"(va),"v"(vb),"v"(idx));
    for (int i=0;i<16;i++) d_out[base*16+i]=vc[i];
}
int main(void){
    const int num_waves=4, threads=num_waves*64;
    uint32_t*h_a=(uint32_t*)malloc((size_t)threads*GA*4);
    uint32_t*h_b=(uint32_t*)malloc((size_t)threads*GB*4);
    float*h_c=(float*)malloc((size_t)threads*CD*4);
    float*h_d=(float*)malloc((size_t)threads*CD*4);
    for(int t=0;t<threads;t++){
        for(int i=0;i<GA;i++) h_a[t*GA+i]=0x3C3C3C3Cu;
        for(int i=0;i<GB;i++) h_b[t*GB+i]=0x38383838u;
        for(int i=0;i<CD;i++)  h_c[t*CD+i]=0;
    }
    uint32_t*d_a,*d_b; float*d_c,*d_d;
    GPU_CHECK(hipMalloc(&d_a,(size_t)threads*GA*4));
    GPU_CHECK(hipMalloc(&d_b,(size_t)threads*GB*4));
    GPU_CHECK(hipMalloc(&d_c,(size_t)threads*CD*4));
    GPU_CHECK(hipMalloc(&d_d,(size_t)threads*CD*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,(size_t)threads*GA*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,(size_t)threads*GB*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,(size_t)threads*CD*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d,0,(size_t)threads*CD*4));
    hipLaunchKernelGGL(smfmac_kernel,dim3(num_waves),dim3(64),0,0,d_a,d_b,d_c,d_d,num_waves);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_d,d_d,(size_t)threads*CD*4,hipMemcpyDeviceToHost));
    const float EXPECTED=32.0f;
    int pass=1;
    for(int t=0;t<threads&&pass;t++) for(int i=0;i<CD;i++){
        if(fabsf(h_d[t*16+i]-EXPECTED)>0.05f*EXPECTED+0.01f){ fprintf(stderr,"FAIL t=%d[%d] got %.3f exp %.3f\n",t,i,h_d[t*16+i],EXPECTED); pass=0; break; }
    }
    printf("v_smfmac_f32_32x32x64_bf8_fp8: exp %.3f\n", EXPECTED);
    printf("%s\n", pass?"PASSED":"FAILED");
    return pass?0:1;
}
