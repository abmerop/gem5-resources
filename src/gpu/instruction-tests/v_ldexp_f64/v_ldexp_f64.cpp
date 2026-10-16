/*
 * HIP test for V_LDEXP_F64 (ENC_VOP3, scale by power of two, double precision).
 * gem5 semantics: D.f64 = S0.f64 * 2^(S1.i).
 * Inline asm forces the VOP3 encoding; S0 is a 64-bit VGPR pair, S1 a 32-bit
 * integer exponent.  Small mantissa and bounded exponent -> result exact.
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
kern(const double *a, const int32_t *e, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s0=a[i], d; int32_t s1=e[i];
    __asm__ volatile("v_ldexp_f64 %0, %1, %2\n\t" : "=v"(d) : "v"(s0), "v"(s1));
    out[i] = d;
}

int main(void)
{
    double *h_a=(double*)malloc(N*8),*h_out=(double*)malloc(N*8);
    int32_t *h_e=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ h_a[i]=(double)((i%101)-50) * 0.25; h_e[i]=(i%41)-20; }
    double *d_a,*d_out; int32_t *d_e;
    GPU_CHECK(hipMalloc(&d_a,N*8));GPU_CHECK(hipMalloc(&d_e,N*4));GPU_CHECK(hipMalloc(&d_out,N*8));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_e,h_e,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_e,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ double exp=ldexp(h_a[i], h_e[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %f exp %f\n",i,h_out[i],exp);pass=0;break;} }
    if(pass)printf("out[1]=%f out[%d]=%f\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_e));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_e);free(h_out);
    return pass?0:1;
}
