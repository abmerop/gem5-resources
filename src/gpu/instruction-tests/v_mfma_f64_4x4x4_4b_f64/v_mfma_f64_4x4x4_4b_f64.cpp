/*
 * HIP test for v_mfma_f64_4x4x4_4b_f64 (VOP3P-MAI matrix instruction,
 * gfx950/CDNA4).  Uniform-input trick: every A elem = 1.0, every B elem = 1.0,
 * C = 0 -> every output = K = 4 (mapping-independent, so it is robust to any
 * layout detail).  Accumulator in AGPR ("+a").
 * Widths (llvm-mc gfx950): A 1 f64 (v[0:1]), B 1 f64, C/D 1 f64 (a[0:1]).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);}} while(0)

static const int CD=1;   /* f64 accumulator elements */

__global__ void mfma_kernel(const double*a_in,const double*b_in,
        const double*c_in,double*d_out,int num_waves) {
    int wf_id=blockIdx.x, lane=threadIdx.x;
    if(wf_id>=num_waves) return;
    int base=wf_id*64+lane;
    double va = a_in[base];
    double vb = b_in[base];
    double vd = c_in[base];
    __asm__ volatile("s_waitcnt vmcnt(0)\n\t"
        "v_mfma_f64_4x4x4_4b_f64 %0, %1, %2, %0\n\t"
        "s_nop 15\n\t s_nop 15\n\t"
        : "+a"(vd) : "v"(va),"v"(vb));
    d_out[base]=vd;
}

int main(void){
    const int num_waves=4, threads=num_waves*64;
    double*h_a=(double*)malloc((size_t)threads*sizeof(double));
    double*h_b=(double*)malloc((size_t)threads*sizeof(double));
    double*h_c=(double*)malloc((size_t)threads*CD*sizeof(double));
    double*h_d=(double*)malloc((size_t)threads*CD*sizeof(double));
    for(int t=0;t<threads;t++){
        h_a[t]=1.0; h_b[t]=1.0;
        for(int i=0;i<CD;i++) h_c[t*CD+i]=0.0;
    }
    double*d_a,*d_b,*d_c,*d_d;
    GPU_CHECK(hipMalloc(&d_a,(size_t)threads*sizeof(double)));
    GPU_CHECK(hipMalloc(&d_b,(size_t)threads*sizeof(double)));
    GPU_CHECK(hipMalloc(&d_c,(size_t)threads*CD*sizeof(double)));
    GPU_CHECK(hipMalloc(&d_d,(size_t)threads*CD*sizeof(double)));
    GPU_CHECK(hipMemcpy(d_a,h_a,(size_t)threads*sizeof(double),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,(size_t)threads*sizeof(double),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,(size_t)threads*CD*sizeof(double),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d,0,(size_t)threads*CD*sizeof(double)));
    hipLaunchKernelGGL(mfma_kernel,dim3(num_waves),dim3(64),0,0,d_a,d_b,d_c,d_d,num_waves);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_d,d_d,(size_t)threads*CD*sizeof(double),hipMemcpyDeviceToHost));
    const double EXPECTED=4.0;
    int pass=1;
    for(int t=0;t<threads&&pass;t++) for(int i=0;i<CD;i++){
        if (fabs(h_d[t*CD+i]-EXPECTED) > 1e-9){ fprintf(stderr,"FAIL t=%d[%d] got %.6f exp %.6f\n",t,i,h_d[t*CD+i],EXPECTED); pass=0; break; }
    }
    printf("v_mfma_f64_4x4x4_4b_f64: out[0]=%.3f exp %.3f\n", h_d[0], EXPECTED);
    printf("%s\n", pass?"PASSED":"FAILED");
    return pass?0:1;
}
