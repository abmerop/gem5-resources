/*
 * HIP test for V_PK_MUL_F16 (ENC_VOP3P, packed f16x2 multiply).
 * gem5 semantics: D[15:0]=S0[15:0]*S1[15:0], D[31:16]=S0[31:16]*S1[31:16] (f16).
 * Inline asm forces the VOP3P encoding; each operand is a packed half2 dword.
 * Small integer inputs -> products exact in f16, bit-exact compare valid.
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
    uint32_t s0=a[i], s1=b[i], d;
    __asm__ volatile("v_pk_mul_f16 %0, %1, %2\n\t" : "=v"(d) : "v"(s0), "v"(s1));
    out[i] = d;
}

static uint32_t pack(int lo,int hi){ __half l=__float2half((float)lo),h=__float2half((float)hi);
    uint16_t lb,hb; __builtin_memcpy(&lb,&l,2); __builtin_memcpy(&hb,&h,2);
    return (uint32_t)lb|((uint32_t)hb<<16); }
/* Multiply in float so IEEE signed-zero (0 * negative = -0.0) matches HW. */
static uint32_t packf(float lo,float hi){ __half l=__float2half(lo),h=__float2half(hi);
    uint16_t lb,hb; __builtin_memcpy(&lb,&l,2); __builtin_memcpy(&hb,&h,2);
    return (uint32_t)lb|((uint32_t)hb<<16); }

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    int *al=(int*)malloc(N*4),*ah=(int*)malloc(N*4),*bl=(int*)malloc(N*4),*bh=(int*)malloc(N*4);
    for(int i=0;i<N;i++){ al[i]=(i%7)-3; ah[i]=(i%5)-2; bl[i]=(i%9)-4; bh[i]=(i%6)-3;
        h_a[i]=pack(al[i],ah[i]); h_b[i]=pack(bl[i],bh[i]); }
    uint32_t *d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t exp=packf((float)al[i]*(float)bl[i], (float)ah[i]*(float)bh[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%08x out[%d]=%08x\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);free(al);free(ah);free(bl);free(bh);
    return pass?0:1;
}
