/*
 * HIP test for v_mfma_f32_4x4x1_16b_f32 (VOP3P-MAI matrix instruction, gfx950/CDNA4).
 * Uniform-input trick: every A elem = 1.0, every B elem = 1.0, C = 0 ->
 * every output = K = 1 (mapping-independent). acc in AGPR ("+a").
 * Widths (llvm-mc gfx950): A 1 VGPR, B 1 VGPR, C/D 4 AGPR.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);}} while(0)
static const int ADW=1, BDW=1, CD=4;
typedef __attribute__((ext_vector_type(4))) float float4_v;
__global__ void mfma_kernel(const uint32_t*a_in,const uint32_t*b_in,
        const float*c_in,float*d_out,int num_waves) {
    int wf_id=blockIdx.x, lane=threadIdx.x;
    if(wf_id>=num_waves) return;
    int base=wf_id*64+lane;
    int va = (int)a_in[base*1];
    int vb = (int)b_in[base*1];
    float4_v vd;
    for (int i=0;i<4;i++) vd[i]=(float)c_in[base*4+i];
    // s_nop after the MFMA: the compiler does NOT insert the mandatory MFMA
    // write -> memory-read hazard waits for an inline-asm MFMA, so the store
    // below can read a[0:3] before the accumulator regs commit (returns the
    // pre-MFMA C value 0 for the later regs on silicon; gem5 has no latency
    // model so it passes either way). 32 waits is safe for this short op.
    __asm__ volatile("s_waitcnt vmcnt(0)\n\t"
        "v_mfma_f32_4x4x1_16b_f32 %0, %1, %2, %0\n\t"
        "s_nop 15\n\t s_nop 15\n\t" : "+a"(vd) : "v"(va),"v"(vb));
    for (int i=0;i<4;i++) d_out[base*4+i]=vd[i];
}
int main(void){
    const int num_waves=4, threads=num_waves*64;
    uint32_t*h_a=(uint32_t*)malloc((size_t)threads*ADW*4);
    uint32_t*h_b=(uint32_t*)malloc((size_t)threads*BDW*4);
    float*h_c=(float*)malloc((size_t)threads*CD*sizeof(float));
    float*h_d=(float*)malloc((size_t)threads*CD*sizeof(float));
    for(int t=0;t<threads;t++){
        for(int i=0;i<ADW;i++) h_a[t*ADW+i]=0x3F800000u;
        for(int i=0;i<BDW;i++) h_b[t*BDW+i]=0x3F800000u;
        for(int i=0;i<CD;i++)  h_c[t*CD+i]=0;
    }
    uint32_t*d_a,*d_b; float*d_c,*d_d;
    GPU_CHECK(hipMalloc(&d_a,(size_t)threads*ADW*4));
    GPU_CHECK(hipMalloc(&d_b,(size_t)threads*BDW*4));
    GPU_CHECK(hipMalloc(&d_c,(size_t)threads*CD*sizeof(float)));
    GPU_CHECK(hipMalloc(&d_d,(size_t)threads*CD*sizeof(float)));
    GPU_CHECK(hipMemcpy(d_a,h_a,(size_t)threads*ADW*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,(size_t)threads*BDW*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,(size_t)threads*CD*sizeof(float),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d,0,(size_t)threads*CD*sizeof(float)));
    hipLaunchKernelGGL(mfma_kernel,dim3(num_waves),dim3(64),0,0,d_a,d_b,d_c,d_d,num_waves);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_d,d_d,(size_t)threads*CD*sizeof(float),hipMemcpyDeviceToHost));
    const float EXPECTED=1.0f;
    int pass=1;
    for(int t=0;t<threads&&pass;t++) for(int i=0;i<CD;i++){
        if (fabsf(h_d[t*4+i]-EXPECTED) > 0.05f*EXPECTED + 0.01f){ fprintf(stderr,"FAIL t=%d[%d] got %.3f exp %.3f\n",t,i,h_d[t*4+i],EXPECTED); pass=0; break; }
    }
    printf("v_mfma_f32_4x4x1_16b_f32: exp %.3f\n", EXPECTED);
    printf("%s\n", pass?"PASSED":"FAILED");
    return pass?0:1;
}
