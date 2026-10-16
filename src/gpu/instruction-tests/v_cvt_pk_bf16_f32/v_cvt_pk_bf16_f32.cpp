/*
 * HIP test for V_CVT_PK_BF16_F32 (ENC_VOP3, packed f32 -> bf16).
 *
 * gem5 semantics (vop3.cc):
 *   tmp0 = bf16(S0.f32); tmp1 = bf16(S1.f32)
 *   D.u32 = (tmp1.bits << 16) | tmp0.bits
 * (no OPSEL/clamp/omod used here.)
 *
 * To make the result independent of the bf16 rounding mode, the f32 inputs are
 * chosen to be *exactly* representable in bf16 (their low 16 mantissa bits are
 * cleared).  For such values bf16(x) == (bits(x) >> 16), bit-exact.
 *
 * Variants: base
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
kern(const float *a, const float *b, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x=a[i], y=b[i]; uint32_t d;
    __asm__ volatile("v_cvt_pk_bf16_f32 %0, %1, %2\n\t" : "=v"(d) : "v"(x), "v"(y));
    out[i]=d;
}

static float frombits(uint32_t b){ float f; __builtin_memcpy(&f,&b,4); return f; }
static uint32_t tobits(float f){ uint32_t b; __builtin_memcpy(&b,&f,4); return b; }

int main(void)
{
    float *h_a=(float*)malloc(N*4), *h_b=(float*)malloc(N*4);
    uint32_t *h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){
        /* build varied signed magnitudes, then clear low 16 bits -> exact bf16 */
        float va=(float)(i-128)*1.5f + 0.25f;
        float vb=(float)(120-i)*0.75f - 0.5f;
        h_a[i]=frombits(tobits(va) & 0xffff0000u);
        h_b[i]=frombits(tobits(vb) & 0xffff0000u);
    }
    float *d_a,*d_b; uint32_t *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N && pass;i++){
        uint32_t bf0=tobits(h_a[i])>>16, bf1=tobits(h_b[i])>>16;
        uint32_t e=(bf1<<16)|bf0;
        if(h_out[i]!=e){fprintf(stderr,"FAIL out[%d] got %08x exp %08x\n",i,h_out[i],e);pass=0;}
    }
    if(pass)printf("out[0]=%08x out[255]=%08x\n",h_out[0],h_out[255]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);
    return pass?0:1;
}
