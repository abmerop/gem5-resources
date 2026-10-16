/*
 * HIP test for V_CMP_F_U64 (ENC_VOP3/VOPC, "F" = always-false 64-bit compare).
 * gem5 semantics: sdst[lane] = 0 for every active lane (constant false).
 * Inline asm forces the VOP3 _e64 encoding; the wave mask goes to an SGPR pair
 * and each lane extracts its own bit (always 0).  Pattern A (vector compare).
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
kern(const uint64_t *a, const uint64_t *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t s0=a[i], s1=b[i], mask;
    __asm__ volatile("v_cmp_f_u64_e64 %0, %1, %2\n\t"
                     : "=s"(mask) : "v"(s0), "v"(s1));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int main(void)
{
    uint64_t *h_a=(uint64_t*)malloc(N*8),*h_b=(uint64_t*)malloc(N*8);
    int *h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){ h_a[i]=(uint64_t)i*2654435761ULL; h_b[i]=(uint64_t)(i^0x5a5a)*40503ULL; }
    uint64_t *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*8));GPU_CHECK(hipMalloc(&d_b,N*8));GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        if(h_out[i]!=0){fprintf(stderr,"FAIL[%d] got %d exp 0\n",i,h_out[i]);pass=0;break;}
    }
    if(pass)printf("out[0]=%d out[%d]=%d (all false)\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);
    return pass?0:1;
}
