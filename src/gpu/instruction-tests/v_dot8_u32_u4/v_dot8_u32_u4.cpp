/*
 * HIP test for V_DOT8_U32_U4 (ENC_VOP3P, 8-way unsigned 4-bit dot product).
 *
 * gem5 (vop3p.cc) computes per lane:
 *   D.u32 = sum_{i=0..7} (S0.nibble[i] * S1.nibble[i]) + S2
 * with no clamp (CLAMP field unset in the plain encoding).
 *
 * Pattern A (vector): per-lane "v" operands force the VOP3P encoding.  All
 * products are 4-bit*4-bit (<=225), sum + S2 wraps mod 2^32 -> exact check.
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
    __asm__ volatile("v_dot8_u32_u4 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_c=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++){
        h_a[i]=(uint32_t)(i*0x12345679u);
        h_b[i]=(uint32_t)(i*0x9e3779b1u + 0x0f0f0f0fu);
        h_c[i]=(uint32_t)(i*3u);
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
        uint32_t sum=h_c[i];
        for(int k=0;k<8;k++){
            uint32_t n0=(h_a[i]>>(4*k))&0xf, n1=(h_b[i]>>(4*k))&0xf;
            sum += n0*n1;
        }
        if(h_out[i]!=sum){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,h_out[i],sum);pass=0;break;}
    }
    if(pass)printf("out[1]=%u out[%d]=%u\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
