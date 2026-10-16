/*
 * HIP test for v_mfma_i32_16x16x64_i8 (VOP3P-MAI matrix instruction, gfx950/CDNA4).
 * Uniform-input trick: every A elem = 1.0, every B elem = 1.0, C = 0 ->
 * every output = K = 64 (mapping-independent). acc in AGPR ("+a").
 * Widths (llvm-mc gfx950): A 4 VGPR, B 4 VGPR, C/D 4 AGPR.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);}} while(0)
static const int ADW=4, BDW=4, CD=4;
typedef __attribute__((ext_vector_type(4))) int int4_v;
__global__ void mfma_kernel(const uint32_t*a_in,const uint32_t*b_in,
        const int*c_in,int*d_out,int num_waves) {
    int wf_id=blockIdx.x, lane=threadIdx.x;
    if(wf_id>=num_waves) return;
    int base=wf_id*64+lane;
    int4_v va;
    for (int i=0;i<4;i++) va[i]=(int)a_in[base*4+i];
    int4_v vb;
    for (int i=0;i<4;i++) vb[i]=(int)b_in[base*4+i];
    int4_v vd;
    for (int i=0;i<4;i++) vd[i]=(int)c_in[base*4+i];
    __asm__ volatile("s_waitcnt vmcnt(0)\n\t"
        "v_mfma_i32_16x16x64_i8 %0, %1, %2, %0\n\t" : "+a"(vd) : "v"(va),"v"(vb));
    for (int i=0;i<4;i++) d_out[base*4+i]=vd[i];
}
int main(void){
    const int num_waves=4, threads=num_waves*64;
    uint32_t*h_a=(uint32_t*)malloc((size_t)threads*ADW*4);
    uint32_t*h_b=(uint32_t*)malloc((size_t)threads*BDW*4);
    int*h_c=(int*)malloc((size_t)threads*CD*sizeof(int));
    int*h_d=(int*)malloc((size_t)threads*CD*sizeof(int));
    for(int t=0;t<threads;t++){
        for(int i=0;i<ADW;i++) h_a[t*ADW+i]=0x01010101u;
        for(int i=0;i<BDW;i++) h_b[t*BDW+i]=0x01010101u;
        for(int i=0;i<CD;i++)  h_c[t*CD+i]=0;
    }
    uint32_t*d_a,*d_b; int*d_c,*d_d;
    GPU_CHECK(hipMalloc(&d_a,(size_t)threads*ADW*4));
    GPU_CHECK(hipMalloc(&d_b,(size_t)threads*BDW*4));
    GPU_CHECK(hipMalloc(&d_c,(size_t)threads*CD*sizeof(int)));
    GPU_CHECK(hipMalloc(&d_d,(size_t)threads*CD*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,(size_t)threads*ADW*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,(size_t)threads*BDW*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,(size_t)threads*CD*sizeof(int),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d,0,(size_t)threads*CD*sizeof(int)));
    hipLaunchKernelGGL(mfma_kernel,dim3(num_waves),dim3(64),0,0,d_a,d_b,d_c,d_d,num_waves);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_d,d_d,(size_t)threads*CD*sizeof(int),hipMemcpyDeviceToHost));
    const int EXPECTED=64;
    int pass=1;
    for(int t=0;t<threads&&pass;t++) for(int i=0;i<CD;i++){
        if (h_d[t*4+i] != EXPECTED){ fprintf(stderr,"FAIL t=%d[%d] got %d exp %d\n",t,i,h_d[t*4+i],EXPECTED); pass=0; break; }
    }
    printf("v_mfma_i32_16x16x64_i8: exp %d\n", EXPECTED);
    printf("%s\n", pass?"PASSED":"FAILED");
    return pass?0:1;
}
