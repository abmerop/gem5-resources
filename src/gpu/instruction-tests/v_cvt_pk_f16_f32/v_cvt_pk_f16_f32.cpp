/*
 * HIP test for V_CVT_PK_F16_F32 (ENC_VOP3, pack-convert two f32 -> two f16).
 * gem5 semantics: D.f16[15:0] = f32_to_f16(S0.f32),
 *                 D.f16[31:16] = f32_to_f16(S1.f32)
 * Inline asm forces the VOP3 V_CVT_PK_F16_F32 encoding.
 * Pattern A (vector): per-lane "v" operands; result is a packed half2 dword.
 * Inputs are small integers (exactly representable in f16) so the conversion
 * is exact and bit-for-bit comparison is valid.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const float *a, const float *b, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s0 = a[i], s1 = b[i];
    uint32_t d;
    __asm__ volatile("v_cvt_pk_f16_f32 %0, %1, %2\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1));
    out[i] = d;
}

int main(void)
{
    float *h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4);
    uint32_t *h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ h_a[i]=(float)((i%37)-18); h_b[i]=(float)((i%29)-14); }
    float *d_a,*d_b; uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        __half lo=__float2half(h_a[i]), hi=__float2half(h_b[i]);
        uint16_t lb,hb; __builtin_memcpy(&lb,&lo,2); __builtin_memcpy(&hb,&hi,2);
        uint32_t exp=(uint32_t)lb | ((uint32_t)hb<<16);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%08x out[%d]=%08x\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);
    return pass?0:1;
}
