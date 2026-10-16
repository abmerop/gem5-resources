/*
 * HIP test for V_CMPX_GT_F32 (VOPC/VOP3 compare-set-EXEC, ordered f32 >).
 *
 *   mask = exec = (S0.f32 > S1.f32)   (ordered: false if either NaN)
 *
 * The _e64 form writes the per-lane mask to an SGPR pair and to EXEC; EXEC is
 * saved/restored around the instruction.  Pattern A (vector), base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cmpx_gt_f32 v_cmpx_gt_f32.cpp
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
kernel_v_cmpx_gt_f32(const float *a, const float *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float va = a[i], vb = b[i];
    uint64_t mask, saved;
    __asm__ volatile(
        "s_mov_b64 %1, exec\n\t"
        "v_cmpx_gt_f32_e64 %0, %2, %3\n\t"
        "s_mov_b64 exec, %1\n\t"
        : "=s"(mask), "=s"(saved) : "v"(va), "v"(vb));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int main(void)
{
    float *h_a=(float*)malloc(N*sizeof(float));
    float *h_b=(float*)malloc(N*sizeof(float));
    int *h_out=(int*)malloc(N*sizeof(int));
    for (int i=0;i<N;i++){
        h_a[i]=(float)(i-128)*0.25f;
        h_b[i]=(float)((i*53)%200 - 100)*0.25f;
        if (i%19==0) h_a[i]=nanf("");
        if (i%29==0) h_b[i]=nanf("");
    }
    float *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(float)));
    GPU_CHECK(hipMalloc(&d_b,N*sizeof(float)));
    GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(float),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(float),hipMemcpyHostToDevice));
    kernel_v_cmpx_gt_f32<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int exp=(h_a[i]>h_b[i])?1:0;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
