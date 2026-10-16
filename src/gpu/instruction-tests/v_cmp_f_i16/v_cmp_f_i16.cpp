/*
 * HIP test for V_CMP_F_I16 (v_cmp_f_i16_e64), CDNA4 / gfx950.
 * Self-checking CMP test with FINITE inputs only (no NaN), so the
 * predicate is unambiguous: result == 0. Each lane extracts its bit
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
kernel_test(const int32_t *a, const int32_t *b, int *out, int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint32_t va=(uint32_t)(a[i]&0xffff), vb=(uint32_t)(b[i]&0xffff);
    uint64_t mask;
    __asm__ volatile("v_cmp_f_i16_e64 %0, %1, %2\n\t":"=s"(mask):"v"(va),"v"(vb));
    out[i]=(int)((mask>>threadIdx.x)&1ULL);
}

int main(void)
{
    int32_t *h_a=(int32_t*)malloc(N*sizeof(int32_t));
    int32_t *h_b=(int32_t*)malloc(N*sizeof(int32_t));
    int *h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){
        switch (i % 6) {
        case 0: h_a[i]=5;     h_b[i]=5;     break;
        case 1: h_a[i]=-3;    h_b[i]=7;     break;
        case 2: h_a[i]=9;     h_b[i]=-2;    break;
        case 3: h_a[i]=-100;  h_b[i]=-50;   break;
        case 4: h_a[i]=42;    h_b[i]=42;    break;
        case 5: h_a[i]=-1;    h_b[i]=-1;    break;
        }
    }
    int32_t *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(int32_t)));GPU_CHECK(hipMalloc(&d_b,N*sizeof(int32_t)));GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(int32_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(int32_t),hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel_test,dim3(N/BLOCK_SIZE),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){
        int16_t av=(int16_t)h_a[i]; int16_t bv=(int16_t)h_b[i];
        int exp=(0)?1:0;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL out[%d] case %d: got %d exp %d\n",i,i%6,h_out[i],exp);pass=0;}
    }
    if(pass)printf("out[0..5]=%d%d%d%d%d%d\n",h_out[0],h_out[1],h_out[2],h_out[3],h_out[4],h_out[5]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out); return pass?0:1;
}
