/*
 * HIP test for V_MADMK_F16 (ENC_VOP2 with inline literal K).
 * gem5 semantics: D.f16 = S0.f16 * K + S1.f16, where K is a 16-bit f16 literal
 * embedded in the instruction.  Here K = 0x4000 = 2.0.
 * Inline asm forces the V_MADMK_F16 encoding with the literal constant.
 * Pattern A (vector): per-lane f16 operands; integer inputs -> exact compare.
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
kern(const uint16_t *a, const uint16_t *b, uint16_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t s0=a[i], s1=b[i], d;
    /* K = 0x4000 (= 2.0 in f16); D = s0 * 2.0 + s1 */
    __asm__ volatile("v_madmk_f16 %0, %1, 0x4000, %2\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1));
    out[i] = d;
}

static uint16_t hb(int v){ __half h=__float2half((float)v); uint16_t b; __builtin_memcpy(&b,&h,2); return b; }

int main(void)
{
    uint16_t *h_a=(uint16_t*)malloc(N*2),*h_b=(uint16_t*)malloc(N*2),*h_out=(uint16_t*)malloc(N*2);
    int *va=(int*)malloc(N*sizeof(int)),*vb=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){ va[i]=(i%13)-6; vb[i]=(i%17)-8; h_a[i]=hb(va[i]); h_b[i]=hb(vb[i]); }
    uint16_t *d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));GPU_CHECK(hipMalloc(&d_out,N*2));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*2,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint16_t exp=hb(va[i]*2+vb[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %04x exp %04x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%04x out[%d]=%04x\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);free(va);free(vb);
    return pass?0:1;
}
