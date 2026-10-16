/*
 * HIP test for V_MAX_F64 (VOP3, double-precision maximum).
 * gem5 (vop3.cc): D.f64 = std::fmax(S0.f64, S1.f64), per active lane.
 * Pattern A (vector): per-lane "v" operands (64-bit -> register pair).
 * No DPP/SDWA (not available for this opcode).
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
kern(const double *a, const double *b, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s0 = a[i], s1 = b[i], d;
    __asm__ volatile("v_max_f64 %0, %1, %2\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1));
    out[i] = d;
}

int main(void)
{
    size_t sz = N * sizeof(double);
    double *h_a=(double*)malloc(sz),*h_b=(double*)malloc(sz),*h_o=(double*)malloc(sz);
    for (int i = 0; i < N; i++) {
        h_a[i] = (double)(i - 128) * 3.5 + 0.125;
        h_b[i] = (double)(64 - i) * 2.25 - 0.5;
    }
    double *d_a,*d_b,*d_o;
    GPU_CHECK(hipMalloc(&d_a,sz));GPU_CHECK(hipMalloc(&d_b,sz));GPU_CHECK(hipMalloc(&d_o,sz));
    GPU_CHECK(hipMemcpy(d_a,h_a,sz,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,sz,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,sz,hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N; i++) {
        double e = fmax(h_a[i], h_b[i]);
        if (h_o[i] != e) { fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,h_o[i],e);pass=0;break; }
    }
    if (pass) printf("out[0]=%g out[%d]=%g\n",h_o[0],N-1,h_o[N-1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_o));
    free(h_a);free(h_b);free(h_o);
    return pass ? 0 : 1;
}
