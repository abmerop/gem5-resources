/*
 * HIP test for V_CMP_F_F16 (ENC_VOP3/VOPC, "F" = always-false f16 compare).
 * gem5 semantics: sdst[lane] = 0 for every active lane (constant false).
 * Inline asm forces the VOP3 _e64 encoding; each lane extracts its own bit.
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
kern(const uint16_t *a, const uint16_t *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t s0=a[i], s1=b[i]; uint64_t mask;
    __asm__ volatile("v_cmp_f_f16_e64 %0, %1, %2\n\t" : "=s"(mask) : "v"(s0), "v"(s1));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

static uint16_t hb(float v){ __half h=__float2half(v); uint16_t b; __builtin_memcpy(&b,&h,2); return b; }

int main(void)
{
    uint16_t *h_a=(uint16_t*)malloc(N*2),*h_b=(uint16_t*)malloc(N*2);
    int *h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){ h_a[i]=hb((float)(i%7)); h_b[i]=hb((float)(i%7)); /* even equal -> still false */ }
    uint16_t *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ if(h_out[i]!=0){fprintf(stderr,"FAIL[%d] got %d exp 0\n",i,h_out[i]);pass=0;break;} }
    if(pass)printf("out[0]=%d out[%d]=%d (all false)\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);
    return pass?0:1;
}
