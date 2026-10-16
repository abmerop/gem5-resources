/*
 * HIP test for V_CMP_GE_U16 (v_cmp_ge_u16_e64), CDNA4 / gfx950.
 * Self-checking CMP test with FINITE inputs only (no NaN), so the
 * predicate is unambiguous: result == (av>=bv). Each lane extracts its bit
 * from the explicit SGPR-pair mask.
 * Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,        \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N=256, BLOCK_SIZE=64;

__global__ void
kernel_test(const uint32_t *a, const uint32_t *b, int *out, int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint32_t va=a[i]&0xffffu, vb=b[i]&0xffffu;
    uint64_t mask;
    __asm__ volatile("v_cmp_ge_u16_e64 %0, %1, %2\n\t":"=s"(mask):"v"(va),"v"(vb));
    out[i]=(int)((mask>>threadIdx.x)&1ULL);
}

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*sizeof(uint32_t));
    uint32_t *h_b=(uint32_t*)malloc(N*sizeof(uint32_t));
    int *h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){
        switch (i % 6) {
        case 0: h_a[i]=5u;     h_b[i]=5u;     break;
        case 1: h_a[i]=10u;    h_b[i]=4000u;  break;
        case 2: h_a[i]=4000u;  h_b[i]=10u;    break;
        case 3: h_a[i]=1u;     h_b[i]=999u;   break;
        case 4: h_a[i]=77u;    h_b[i]=77u;    break;
        case 5: h_a[i]=123u;   h_b[i]=123u;   break;
        }
    }
    uint32_t *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(uint32_t)));GPU_CHECK(hipMalloc(&d_b,N*sizeof(uint32_t)));GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(uint32_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(uint32_t),hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel_test,dim3(N/BLOCK_SIZE),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){
        uint16_t av=(uint16_t)h_a[i]; uint16_t bv=(uint16_t)h_b[i];
        int exp=((av>=bv))?1:0;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL out[%d] case %d: got %d exp %d\n",i,i%6,h_out[i],exp);pass=0;}
    }
    if(pass)printf("out[0..5]=%d%d%d%d%d%d\n",h_out[0],h_out[1],h_out[2],h_out[3],h_out[4],h_out[5]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out); return pass?0:1;
}
