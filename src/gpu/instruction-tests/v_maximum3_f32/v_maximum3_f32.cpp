/*
 * HIP test for V_MAXIMUM3_F32 (VOP3, IEEE 3-input maximum, NaN-propagating).
 * gem5 (vop3.cc): D = maxPropNan(maxPropNan(S0,S1),S2). With finite inputs
 * (no NaN) this is simply std::max(std::max(S0,S1),S2), independent of the
 * dx10_clamp mode. This test uses only finite values.
 * Pattern A (vector): three "v" inputs. No DPP/SDWA (not supported).
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
    float s0 = a[i], s1 = b[i], s2 = c[i], d;
    __asm__ volatile("v_maximum3_f32 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

int main(void)
{
    size_t sz = N * sizeof(float);
    float *h_a=(float*)malloc(sz),*h_b=(float*)malloc(sz),
          *h_c=(float*)malloc(sz),*h_o=(float*)malloc(sz);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)((i * 37) % 211) - 105.0f + 0.5f;
        h_b[i] = (float)((i * 53) % 199) - 99.0f - 0.25f;
        h_c[i] = (float)((i * 71) % 233) - 116.0f + 0.125f;
    }
    float *d_a,*d_b,*d_c,*d_o;
    GPU_CHECK(hipMalloc(&d_a,sz));GPU_CHECK(hipMalloc(&d_b,sz));
    GPU_CHECK(hipMalloc(&d_c,sz));GPU_CHECK(hipMalloc(&d_o,sz));
    GPU_CHECK(hipMemcpy(d_a,h_a,sz,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,sz,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,sz,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,sz,hipMemcpyDeviceToHost));
    int pass = 1;
    for (int i = 0; i < N; i++) {
        float m = h_a[i] > h_b[i] ? h_a[i] : h_b[i];
        float e = m > h_c[i] ? m : h_c[i];
        if (h_o[i] != e) { fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,h_o[i],e);pass=0;break; }
    }
    if (pass) printf("out[0]=%g out[%d]=%g\n",h_o[0],N-1,h_o[N-1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_o));
    free(h_a);free(h_b);free(h_c);free(h_o);
    return pass ? 0 : 1;
}
