/*
 * HIP test for V_MUL_LEGACY_F32 (ENC_VOP3, DX9 multiply).
 * gem5 semantics: D.f = S0.f * S1.f (DX9 rules, 0.0*x = 0.0).
 * For nonzero finite operands this equals the ordinary product.
 * Inline asm forces the VOP3 encoding; small integer inputs -> product exact.
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
kern(const float *a, const float *b, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s0=a[i], s1=b[i], d;
    __asm__ volatile("v_mul_legacy_f32 %0, %1, %2\n\t" : "=v"(d) : "v"(s0), "v"(s1));
    out[i] = d;
}

int main(void)
{
    float *h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4),*h_out=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ h_a[i]=(float)((i%63)-31) + 1.0f; h_b[i]=(float)((i%47)-23) - 1.0f; }
    float *d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ float exp=h_a[i]*h_b[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %f exp %f\n",i,h_out[i],exp);pass=0;break;} }
    if(pass)printf("out[1]=%f out[%d]=%f\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);
    return pass?0:1;
}
