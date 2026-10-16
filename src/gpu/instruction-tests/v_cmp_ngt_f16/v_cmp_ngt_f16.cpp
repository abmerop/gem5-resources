/*
 * HIP test for V_CMP_NGT_F16 (VOP3 f16 not-greater-than).
 * gem5 (vop3_cmp.cc): lane mask bit = !(a > b) on f16 values (computed in
 * float).  Inputs are non-NaN f16 so !(a>b) == (a<=b).  _e64 form -> SGPR mask.
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
kern(const _Float16 *a, const _Float16 *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    _Float16 va = a[i], vb = b[i]; uint64_t mask;
    __asm__ volatile("v_cmp_ngt_f16_e64 %0, %1, %2\n\t"
                     : "=s"(mask) : "v"(va), "v"(vb));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int main(void)
{
    _Float16 *h_a=(_Float16*)malloc(N*sizeof(_Float16));
    _Float16 *h_b=(_Float16*)malloc(N*sizeof(_Float16));
    int *h_out=(int*)malloc(N*sizeof(int));
    for (int i=0;i<N;i++){ h_a[i]=(_Float16)((float)(i%7)-3.0f); h_b[i]=(_Float16)((float)(i%5)-2.0f); }
    _Float16 *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(_Float16)));GPU_CHECK(hipMalloc(&d_b,N*sizeof(_Float16)));
    GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(_Float16),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(_Float16),hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++){
        int exp = !((float)h_a[i] > (float)h_b[i]) ? 1 : 0;
        if (h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}
    }
    if (pass) printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
