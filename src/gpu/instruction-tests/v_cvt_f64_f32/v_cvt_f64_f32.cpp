/*
 * HIP test for V_CVT_F64_F32 (ENC_VOP1, widen f32 -> f64).
 * gem5 semantics: D.f64 = (double)S0.f32.
 * Inline asm forces the VOP1 encoding; 64-bit VGPR-pair destination.
 * f32 -> f64 is always exact -> bit-exact comparison valid for any input.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const float *a, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s0=a[i]; double d;
    __asm__ volatile("v_cvt_f64_f32 %0, %1\n\t" : "=v"(d) : "v"(s0));
    out[i] = d;
}

int main(void)
{
    float *h_a=(float*)malloc(N*4); double *h_out=(double*)malloc(N*8);
    for(int i=0;i<N;i++) h_a[i]=(float)((i*7)%1000-500) * 0.125f + (float)i*1.5f;
    float *d_a; double *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_out,N*8));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        double exp=(double)h_a[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %f exp %f\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%f out[%d]=%f\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_out);
    return pass?0:1;
}
