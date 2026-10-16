/*
 * HIP test for V_CMP_NE_I16 (ENC_VOP3/VOPC, 16-bit not-equal compare).
 * gem5 semantics: sdst[lane] = (S0.i16 != S1.i16) ? 1 : 0  (per-lane mask).
 * Inline asm forces the VOP3 V_CMP_NE_I16_e64 encoding; the wave mask is read
 * into an explicit SGPR pair and each lane extracts its own bit.
 * Pattern A (vector compare): "v" 16-bit operands, "=s" mask.
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
kern(const uint16_t *a, const uint16_t *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t s0=a[i], s1=b[i];
    uint64_t mask;
    __asm__ volatile("v_cmp_ne_i16_e64 %0, %1, %2\n\t"
                     : "=s"(mask) : "v"(s0), "v"(s1));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int main(void)
{
    uint16_t *h_a=(uint16_t*)malloc(N*2),*h_b=(uint16_t*)malloc(N*2);
    int *h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){
        h_a[i]=(uint16_t)(i*131+7);
        h_b[i]=(uint16_t)((i%3==0)? (i*131+7) : (i*977+1)); /* some equal */
    }
    uint16_t *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int exp=(h_a[i]!=h_b[i])?1:0;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);
    return pass?0:1;
}
