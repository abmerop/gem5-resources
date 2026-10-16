/*
 * HIP test for V_PK_FMA_F16 (ENC_VOP3P, packed half-precision FMA).
 *
 *   D[half] = S0[half] * S1[half] + S2[half]   (per packed 16-bit lane)
 *
 * gem5 vop3p.cc computes fplibMulAdd(S2, S0, S1) = S2 + S0*S1.  All operands
 * are small exactly-representable integer-valued halves so the FMA result is
 * exact and bit-checkable.  Pattern A (vector), base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_pk_fma_f16 v_pk_fma_f16.cpp
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
kernel(const uint32_t *a, const uint32_t *b, const uint32_t *c,
       uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t va=a[i], vb=b[i], vc=c[i], vd;
    __asm__ volatile("v_pk_fma_f16 %0, %1, %2, %3\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb), "v"(vc));
    out[i] = vd;
}

static uint16_t hb(__half h){ uint16_t b; __builtin_memcpy(&b,&h,2); return b; }
static __half mh(int v){ return __float2half((float)v); }

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_c=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    int A0[N],A1[N],B0[N],B1[N],C0[N],C1[N];
    for(int i=0;i<N;i++){
        A0[i]=(i%7)-3; A1[i]=(i%5)-2; B0[i]=(i%6)-2; B1[i]=(i%9)-4;
        C0[i]=(i%11)-5; C1[i]=(i%4)-1;
        h_a[i]=((uint32_t)hb(mh(A1[i]))<<16)|hb(mh(A0[i]));
        h_b[i]=((uint32_t)hb(mh(B1[i]))<<16)|hb(mh(B0[i]));
        h_c[i]=((uint32_t)hb(mh(C1[i]))<<16)|hb(mh(C0[i]));
    }
    uint32_t *d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint16_t lo=hb(mh(A0[i]*B0[i]+C0[i])), hi=hb(mh(A1[i]*B1[i]+C1[i]));
        uint32_t exp=((uint32_t)hi<<16)|lo;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
