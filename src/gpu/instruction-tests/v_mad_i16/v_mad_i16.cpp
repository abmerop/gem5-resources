/*
 * HIP test for V_MAD_I16 (ENC_VOP3, signed 16-bit multiply-add).
 * gem5 semantics: D.i16 = S0.i16 * S1.i16 + S2.i16  (low 16 bits).
 * Inline asm forces the VOP3 encoding; per-lane 16-bit operands.
 * Output read as int16_t. Exact modular-16 compare.
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
kern(const int16_t *a, const int16_t *b, const int16_t *c, int16_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int16_t s0=a[i], s1=b[i], s2=c[i], d;
    __asm__ volatile("v_mad_i16 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

int main(void)
{
    int16_t *h_a=(int16_t*)malloc(N*2),*h_b=(int16_t*)malloc(N*2),*h_c=(int16_t*)malloc(N*2),*h_out=(int16_t*)malloc(N*2);
    for(int i=0;i<N;i++){ h_a[i]=(int16_t)((i*7)%61-30); h_b[i]=(int16_t)((i*11)%41-20); h_c[i]=(int16_t)((i*5)%201-100); }
    int16_t *d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));GPU_CHECK(hipMalloc(&d_c,N*2));GPU_CHECK(hipMalloc(&d_out,N*2));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*2,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*2,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int16_t exp=(int16_t)(h_a[i]*h_b[i]+h_c[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%d out[%d]=%d\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
