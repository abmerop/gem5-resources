/*
 * HIP test for V_PK_ASHRREV_I16 (ENC_VOP3P, packed 2x16 arithmetic shift right,
 * reversed operands).  Decodes to Inst_VOP3P__V_PK_ASHRREV_B16 in gem5, whose
 * per-half op (vop3p.cc) is:
 *   D.i16[h] = (int16_t)( (int32_t)S1.i16[h] >> (S0.i16[h] & 0xf) )
 * i.e. S0 supplies the shift amount, S1 the (signed) data.  Base only.
 *
 * Pattern A: per-lane "v" operands force the VOP3P encoding.  Data includes
 * negatives to exercise the arithmetic (sign-extending) shift; exact check.
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
kern(const uint32_t *a, const uint32_t *b, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s0 = a[i], s1 = b[i], d;
    __asm__ volatile("v_pk_ashrrev_i16 %0, %1, %2\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1));
    out[i] = d;
}

static inline uint16_t ashr_half(int16_t data, int16_t sh)
{
    int32_t d = data;
    return (uint16_t)(int16_t)(d >> (sh & 0xf));
}

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++){
        int16_t slo=(int16_t)(i%16), shi=(int16_t)((i/2)%16);
        h_a[i]=((uint16_t)shi<<16)|(uint16_t)slo;                 /* shifts */
        int16_t dlo=(int16_t)(-30000 + i*233);
        int16_t dhi=(int16_t)( 20000 - i*191);
        h_b[i]=((uint16_t)dhi<<16)|(uint16_t)dlo;                 /* data   */
    }
    uint32_t *d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int16_t s0lo=(int16_t)(h_a[i]&0xffff), s0hi=(int16_t)(h_a[i]>>16);
        int16_t s1lo=(int16_t)(h_b[i]&0xffff), s1hi=(int16_t)(h_b[i]>>16);
        uint32_t exp=((uint32_t)ashr_half(s1hi,s0hi)<<16)|ashr_half(s1lo,s0lo);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=0x%08x out[%d]=0x%08x\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);
    return pass?0:1;
}
