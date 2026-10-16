/*
 * HIP test for V_CMP_GT_F64 (VCC mask bit = (S0.f64 > S1.f64); exact halves, no NaN).
 * gem5: src/arch/amdgpu/common/insts/vop3_cmp.cc. The _e64 form writes the
 * per-lane result to an SGPR mask; bit (threadIdx.x) is the lane's result.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                          \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const double *a, const double *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double A_ = a[i], B_ = b[i];
    double va = A_;
    double vb = B_;
    uint64_t mask;
    __asm__ volatile(
        "v_cmp_gt_f64_e64 %0, %1, %2\n\t"
        : "=s"(mask)
        : "v"(va), "v"(vb));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int main(void)
{
    double *h_a=(double*)malloc(N*sizeof(double));
    double *h_b=(double*)malloc(N*sizeof(double));
    int *h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){ h_a[i]=(double)((double)((i%11)-5)*0.25); h_b[i]=(double)((double)((i%9)-4)*0.5); }
    double *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(double)));
    GPU_CHECK(hipMalloc(&d_b,N*sizeof(double)));
    GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(double),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(double),hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1,ntrue=0;
    for(int i=0;i<N;i++){
        double A=h_a[i], B=h_b[i];
        int exp=(A > B)?1:0; ntrue+=exp;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("true=%d/%d\n",ntrue,N);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);
    return pass?0:1;
}
