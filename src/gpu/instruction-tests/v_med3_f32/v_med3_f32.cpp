/*
 * HIP test for V_MED3_F32 (ENC_VOP3, median of three f32).
 * gem5 semantics: D.f32 = median(S0.f32, S1.f32, S2.f32)
 * Inline asm forces the VOP3 V_MED3_F32 encoding.
 * Pattern A (vector): per-lane "v" f32 operands.
 * Distinct integer-valued f32 inputs -> median is well defined and exact.
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
kern(const float *a, const float *b, const float *c, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s0=a[i], s1=b[i], s2=c[i], d;
    __asm__ volatile("v_med3_f32 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

static float med3(float x,float y,float z){
    float mn=x<y?x:y, mx=x<y?y:x;
    if(z<mn) return mn;
    if(z>mx) return mx;
    return z;
}

int main(void)
{
    float *h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4),*h_c=(float*)malloc(N*4),*h_out=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ h_a[i]=(float)((i*7)%101-50); h_b[i]=(float)((i*13)%97-48); h_c[i]=(float)((i*5)%89-44); }
    float *d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        float exp=med3(h_a[i],h_b[i],h_c[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %f exp %f\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%f out[%d]=%f\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
