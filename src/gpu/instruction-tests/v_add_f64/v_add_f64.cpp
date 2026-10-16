/*
 * HIP test for V_ADD_F64 (VOP3 double-precision add).
 * gem5 (vop3.cc): for normal finite non-zero operands, vdst = src0 + src1.
 * Inputs are small integer-valued doubles (exactly representable), avoiding
 * the zero/subnormal/inf/nan special-case paths, so compared bit-exactly.
 * Variants: base.
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
kern(const double *a, const double *b, double *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    double va = a[i], vb = b[i], vd;
    __asm__ volatile("v_add_f64 %0, %1, %2\n\t" : "=v"(vd) : "v"(va), "v"(vb));
    out[i] = vd;
}

int main(void)
{
    double *h_a=(double*)malloc(N*8),*h_b=(double*)malloc(N*8),*h_o=(double*)malloc(N*8);
    for (int i=0;i<N;i++){ h_a[i]=(double)((i%97)-48)+0.5; h_b[i]=(double)((i%61)-30)*4.0+0.25; }
    double *d_a,*d_b,*d_o;
    GPU_CHECK(hipMalloc(&d_a,N*8));GPU_CHECK(hipMalloc(&d_b,N*8));GPU_CHECK(hipMalloc(&d_o,N*8));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++){
        double exp = h_a[i] + h_b[i];
        if (h_o[i]!=exp){fprintf(stderr,"FAIL[%d] got %.17g exp %.17g\n",i,h_o[i],exp);pass=0;break;}
    }
    if (pass) printf("out[0]=%.17g out[%d]=%.17g\n",h_o[0],N-1,h_o[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
