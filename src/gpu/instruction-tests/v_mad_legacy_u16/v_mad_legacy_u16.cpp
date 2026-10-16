/*
 * HIP test for V_MAD_LEGACY_U16 (VOP3, 16-bit unsigned multiply-add).
 *   D.u16 = (S0.u16 * S1.u16 + S2.u16) mod 2^16
 * Legacy form has identical integer semantics to v_mad_u16; explicit inline
 * asm forces the VOP3 encoding.  Pattern A (per-lane "v" operands).  Base only.
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
    uint32_t s0=a[i], s1=b[i], s2=c[i], d;
    __asm__ volatile("v_mad_legacy_u16 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = (uint16_t)d;
}

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_c=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++){
        h_a[i]=(uint16_t)(i*131u+7u);
        h_b[i]=(uint16_t)(i*17u+3u);
        h_c[i]=(uint16_t)(i*911u+1u);
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
        uint16_t exp=(uint16_t)((uint16_t)h_a[i]*(uint16_t)h_b[i]+(uint16_t)h_c[i]);
        if((uint16_t)h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",
            i,(uint16_t)h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%u out[%d]=%u\n",(uint16_t)h_out[1],N-1,(uint16_t)h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
