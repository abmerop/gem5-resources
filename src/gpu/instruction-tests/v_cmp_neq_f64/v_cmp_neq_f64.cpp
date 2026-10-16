/*
 * HIP test for V_CMP_NEQ_F64 (VOPC/VOP3 compare, f64 not-equal).
 *
 *   mask[lane] = (S0.f64 != S1.f64)   (true also when either is NaN)
 *
 * Matches gem5 vopc.cc (src0 != src1 in C semantics).  Pattern A, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cmp_neq_f64 v_cmp_neq_f64.cpp
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
kernel(const double *a, const double *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double va=a[i], vb=b[i];
    uint64_t mask;
    __asm__ volatile("v_cmp_neq_f64_e64 %0, %1, %2\n\t" : "=s"(mask) : "v"(va), "v"(vb));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int main(void)
{
    double *h_a=(double*)malloc(N*8),*h_b=(double*)malloc(N*8);
    int *h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){
        h_a[i]=(double)(i%11-5)*0.5;
        h_b[i]=(double)(i%9-4)*0.5;
        if(i%6==0) h_b[i]=h_a[i];
        if(i%17==0) h_a[i]=nan("");
        if(i%23==0) h_b[i]=nan("");
    }
    double *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*8));GPU_CHECK(hipMalloc(&d_b,N*8));GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ int exp=(h_a[i]!=h_b[i])?1:0; if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;} }
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
