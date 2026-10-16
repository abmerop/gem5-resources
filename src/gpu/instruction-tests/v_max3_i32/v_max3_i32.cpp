/*
 * HIP test for V_MAX3_I32 (ENC_VOP3, signed maximum of three).
 * gem5: D.i32 = max(max(S0,S1),S2)  (signed). vop3.cc.
 * Pattern A (vector): per-lane "v" operands; exact integer compare.
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
kern(const int32_t *a, const int32_t *b, const int32_t *c, int32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t s0 = a[i], s1 = b[i], s2 = c[i], d;
    __asm__ volatile("v_max3_i32 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

int main(void)
{
    int32_t *h_a=(int32_t*)malloc(N*4),*h_b=(int32_t*)malloc(N*4);
    int32_t *h_c=(int32_t*)malloc(N*4),*h_out=(int32_t*)malloc(N*4);
    for (int i=0;i<N;i++){
        h_a[i]=(int32_t)(i*1103515245 + 12345);
        h_b[i]=(int32_t)(-i*2654435761u + 99);
        h_c[i]=(int32_t)((i-128)*16777259);
    }
    int32_t *d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int32_t m=h_a[i]>h_b[i]?h_a[i]:h_b[i];
        int32_t exp=m>h_c[i]?m:h_c[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%d out[%d]=%d\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
