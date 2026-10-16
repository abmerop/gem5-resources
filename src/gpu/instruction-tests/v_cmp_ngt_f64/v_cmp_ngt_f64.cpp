/*
 * HIP test for V_CMP_NGT_F64 (ENC_VOP3/VOPC, "not greater-than" f64 compare).
 * gem5 semantics: sdst[lane] = !(S0.f64 > S1.f64) ? 1 : 0  (true also if NaN).
 * Inline asm forces the VOP3 _e64 encoding; each lane extracts its own bit.
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
kern(const double *a, const double *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s0=a[i], s1=b[i]; uint64_t mask;
    __asm__ volatile("v_cmp_ngt_f64_e64 %0, %1, %2\n\t" : "=s"(mask) : "v"(s0), "v"(s1));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int main(void)
{
    double *h_a=(double*)malloc(N*8),*h_b=(double*)malloc(N*8);
    int *h_out=(int*)malloc(N*sizeof(int));
    double qnan=nan("");
    for(int i=0;i<N;i++){
        switch(i%4){
        case 0: h_a[i]=(double)(i%9); h_b[i]=(double)(i%5); break;   /* gt mix */
        case 1: h_a[i]=(double)(i%5); h_b[i]=(double)(i%5)+1.0; break;/* not gt */
        case 2: h_a[i]=qnan;          h_b[i]=1.0; break;             /* NaN -> true */
        case 3: h_a[i]=3.0;           h_b[i]=3.0; break;             /* eq -> not gt */
        }
    }
    double *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*8));GPU_CHECK(hipMalloc(&d_b,N*8));GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ int exp=!(h_a[i]>h_b[i])?1:0;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;} }
    if(pass)printf("out[0]=%d out[2]=%d out[%d]=%d\n",h_out[0],h_out[2],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);
    return pass?0:1;
}
