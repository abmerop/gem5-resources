/*
 * HIP test for V_PK_ADD_F16 (ENC_VOP3P, packed f16x2 add).
 * gem5 semantics: D[15:0]  = S0[15:0]  + S1[15:0]   (f16)
 *                 D[31:16] = S0[31:16] + S1[31:16]  (f16)
 * Inline asm forces the VOP3P V_PK_ADD_F16 encoding.
 * Pattern A (vector): each operand is a packed half2 carried in a uint32_t.
 * Inputs are small integers (exact in f16, sums exact) -> bit-exact compare.
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
kern(const uint32_t *a, const uint32_t *b, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s0 = a[i], s1 = b[i], d;
    __asm__ volatile("v_pk_add_f16 %0, %1, %2\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1));
    out[i] = d;
}

static uint32_t pack(int lo, int hi){
    __half l=__float2half((float)lo), h=__float2half((float)hi);
    uint16_t lb,hb; __builtin_memcpy(&lb,&l,2); __builtin_memcpy(&hb,&h,2);
    return (uint32_t)lb | ((uint32_t)hb<<16);
}

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    int *alo=(int*)malloc(N*sizeof(int)),*ahi=(int*)malloc(N*sizeof(int));
    int *blo=(int*)malloc(N*sizeof(int)),*bhi=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){
        alo[i]=(i%21)-10; ahi[i]=(i%17)-8;
        blo[i]=(i%13)-6;  bhi[i]=(i%11)-5;
        h_a[i]=pack(alo[i],ahi[i]); h_b[i]=pack(blo[i],bhi[i]);
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
        uint32_t exp=pack(alo[i]+blo[i], ahi[i]+bhi[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%08x out[%d]=%08x\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);free(alo);free(ahi);free(blo);free(bhi);
    return pass?0:1;
}
