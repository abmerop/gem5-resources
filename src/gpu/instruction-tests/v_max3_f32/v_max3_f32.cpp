/*
 * HIP test for V_MAX3_F32 (ENC_VOP3, maximum of three f32).
 * gem5 (vop3.cc): D = fmax(fmax(S0,S1),S2).
 * All inputs are exactly-representable f32 (multiples of 0.25), no NaN/Inf, so
 * the result is bit-identical to one of the operands.
 * Pattern A (vector): per-lane "v" operands force the VOP3 encoding.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
    float s0 = a[i], s1 = b[i], s2 = c[i], d;
    __asm__ volatile("v_max3_f32 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

int main(void)
{
    float *h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4);
    float *h_c=(float*)malloc(N*4),*h_out=(float*)malloc(N*4);
    for (int i=0;i<N;i++){
        h_a[i]=(float)((i%23)-11)*0.25f;
        h_b[i]=(float)((i%17)-8)*0.5f;
        h_c[i]=(float)((i%13)-6)*0.75f;
    }
    float *d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        float exp=fmaxf(fmaxf(h_a[i],h_b[i]),h_c[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%g out[%d]=%g\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
