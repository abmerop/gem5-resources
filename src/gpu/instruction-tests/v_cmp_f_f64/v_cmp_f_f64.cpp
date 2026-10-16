/*
 * HIP test for V_CMP_F_F64 (VOPC/VOP3 compare, "F" = always false, f64).
 *
 *   mask[lane] = 0   (unconditionally)
 *
 * The _e64 form writes the per-lane mask to an SGPR pair; every lane bit must
 * read back 0 regardless of operand values.  Pattern A (vector), base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cmp_f_f64 v_cmp_f_f64.cpp
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
kernel_v_cmp_f_f64(const double *a, const double *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double va = a[i], vb = b[i];
    uint64_t mask;
    __asm__ volatile("v_cmp_f_f64_e64 %0, %1, %2\n\t"
                     : "=s"(mask) : "v"(va), "v"(vb));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int main(void)
{
    double *h_a=(double*)malloc(N*sizeof(double));
    double *h_b=(double*)malloc(N*sizeof(double));
    int *h_out=(int*)malloc(N*sizeof(int));
    for (int i=0;i<N;i++){ h_a[i]=(double)(i-128)*1.25; h_b[i]=(double)(i-128)*1.25; }
    double *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(double)));
    GPU_CHECK(hipMalloc(&d_b,N*sizeof(double)));
    GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(double),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(double),hipMemcpyHostToDevice));
    kernel_v_cmp_f_f64<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        if(h_out[i]!=0){fprintf(stderr,"FAIL[%d] got %d exp 0\n",i,h_out[i]);pass=0;break;}
    }
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
