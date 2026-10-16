/*
 * HIP test for V_SAD_HI_U8 (ENC_VOP3, per-byte difference sum into high half).
 *
 * gem5 (vop3.cc) computes per lane (note: it uses raw byte subtraction, NOT
 * absolute value):
 *   D.u32 = (((b3-b3') + (b2-b2') + (b1-b1') + (b0-b0')) << 16) + S2
 * where bN/bN' are the corresponding bytes of S0/S1.
 *
 * To make the result independent of signedness/wrap quirks, every byte of S0
 * is chosen >= the matching byte of S1, so each difference is a non-negative
 * 0..255 value.  The host replicates the exact uint32 expression.
 *
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
kern(const uint32_t *a, const uint32_t *b, const uint32_t *c,
     uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s0 = a[i], s1 = b[i], s2 = c[i], d;
    __asm__ volatile("v_sad_hi_u8 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

static inline uint32_t byte(uint32_t v, int n){ return (v >> (8*n)) & 0xff; }

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_c=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++){
        /* build S1 bytes, then S0 byte >= S1 byte */
        uint32_t b0=(i*7)&0x7f, b1=(i*13)&0x7f, b2=(i*5)&0x7f, b3=(i*3)&0x7f;
        uint32_t a0=b0+((i*2)&0x7f), a1=b1+((i)&0x7f);
        uint32_t a2=b2+((i*4)&0x7f), a3=b3+((i*6)&0x7f);
        h_b[i]=(b3<<24)|(b2<<16)|(b1<<8)|b0;
        h_a[i]=((a3&0xff)<<24)|((a2&0xff)<<16)|((a1&0xff)<<8)|(a0&0xff);
        h_c[i]=(uint32_t)(i*0x9e3779b1u)+7u;
    }
    uint32_t *d_a,*d_b,*d_c,*d_out;
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
        uint32_t s=(byte(h_a[i],3)-byte(h_b[i],3))+(byte(h_a[i],2)-byte(h_b[i],2))
                  +(byte(h_a[i],1)-byte(h_b[i],1))+(byte(h_a[i],0)-byte(h_b[i],0));
        uint32_t exp=(s<<16)+h_c[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%08x out[%d]=%08x\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
