/*
 * HIP test for V_CMP_NLG_F32 (VOP3 f32 not-less-greater = "equal or unordered").
 * gem5 (vop3_cmp.cc): lane mask bit = !(a < b || a > b).  For non-NaN inputs
 * this is exactly (a == b).  _e64 form -> SGPR mask.  Variants: base.
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
kern(const float *a, const float *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float va = a[i], vb = b[i]; uint64_t mask;
    __asm__ volatile("v_cmp_nlg_f32_e64 %0, %1, %2\n\t"
                     : "=s"(mask) : "v"(va), "v"(vb));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int main(void)
{
    float *h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4);
    int *h_out=(int*)malloc(N*sizeof(int));
    for (int i=0;i<N;i++){
        h_a[i]=(float)((i%5)-2);
        /* half the lanes equal, half differ */
        h_b[i]=(i%2)?(float)((i%5)-2):(float)((i%7)-3);
    }
    float *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++){
        int exp = !(h_a[i] < h_b[i] || h_a[i] > h_b[i]) ? 1 : 0;
        if (h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}
    }
    if (pass) printf("out[0]=%d out[1]=%d\n",h_out[0],h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
