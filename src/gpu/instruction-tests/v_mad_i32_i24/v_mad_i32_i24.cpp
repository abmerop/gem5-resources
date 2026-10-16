/*
 * HIP test for V_MAD_I32_I24 (ENC_VOP3, signed 24-bit multiply-add).
 * gem5 (vop3.cc): D.i32 = sext24(S0[23:0]) * sext24(S1[23:0]) + S2.i32.
 * Pattern A (vector): per-lane "v" operands force the VOP3 encoding.
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
    __asm__ volatile("v_mad_i32_i24 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

static int32_t sext24(uint32_t v){ v &= 0xffffff; if (v & 0x800000) v |= 0xff000000u; return (int32_t)v; }

int main(void)
{
    int32_t *h_a=(int32_t*)malloc(N*4),*h_b=(int32_t*)malloc(N*4);
    int32_t *h_c=(int32_t*)malloc(N*4),*h_out=(int32_t*)malloc(N*4);
    for (int i=0;i<N;i++){
        h_a[i]=(int32_t)((i-128)*4099);     /* fits in 24-bit signed range */
        h_b[i]=(int32_t)((64-i)*1301);
        h_c[i]=(int32_t)(i*131071 - 1000000);
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
        int32_t exp=(int32_t)((uint32_t)(sext24(h_a[i])*sext24(h_b[i])) + (uint32_t)h_c[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%d out[%d]=%d\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
