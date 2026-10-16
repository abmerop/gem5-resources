/*
 * HIP test for V_CMP_NE_U16 (VOPC u16 not-equal).
 * gem5 (vopc.cc): lane mask bit = (S0.u16 != S1.u16) on low 16 bits.
 * _e64 form -> SGPR mask.  Variants: base.
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
kern(const uint32_t *a, const uint32_t *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint32_t va = a[i], vb = b[i]; uint64_t mask = 0;
    __asm__ volatile("v_cmp_ne_u16_e64 %0, %1, %2\n\t"
                     : "=s"(mask) : "v"(va), "v"(vb));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    int *h_out=(int*)malloc(N*sizeof(int));
    for (int i=0;i<N;i++){ h_a[i]=(uint32_t)(uint16_t)(i%7); h_b[i]=(uint32_t)(uint16_t)((i%2)?(i%7):(i%5)); }
    uint32_t *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++){
        int exp = ((uint16_t)h_a[i] != (uint16_t)h_b[i]) ? 1 : 0;
        if (h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}
    }
    if (pass) printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
