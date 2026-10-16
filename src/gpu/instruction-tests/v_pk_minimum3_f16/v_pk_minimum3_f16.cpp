/*
 * HIP test for V_PK_MINIMUM3_F16 (ENC_VOP3P, packed IEEE minimum3 of f16x2).
 * gem5 semantics, per packed half: D = minimum(S0, minimum(S1, S2)).
 * For non-NaN inputs this is the ordinary 3-way min.
 * Inline asm forces the VOP3P encoding; each operand is a packed half2 dword.
 * Integer-valued inputs -> exact bit compare.
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
kern(const uint32_t *a, const uint32_t *b, const uint32_t *c, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s0=a[i], s1=b[i], s2=c[i], d;
    __asm__ volatile("v_pk_minimum3_f16 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

static uint32_t pack(int lo,int hi){ __half l=__float2half((float)lo),h=__float2half((float)hi);
    uint16_t lb,hb; __builtin_memcpy(&lb,&l,2); __builtin_memcpy(&hb,&h,2);
    return (uint32_t)lb|((uint32_t)hb<<16); }
static int mn3(int a,int b,int c){int m=a;if(b<m)m=b;if(c<m)m=c;return m;}

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4),*h_c=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    int *al=(int*)malloc(N*4),*ah=(int*)malloc(N*4),*bl=(int*)malloc(N*4),*bh=(int*)malloc(N*4),*cl=(int*)malloc(N*4),*ch=(int*)malloc(N*4);
    for(int i=0;i<N;i++){ al[i]=(i%21)-10; ah[i]=(i%17)-8; bl[i]=(i%13)-6; bh[i]=(i%11)-5; cl[i]=(i%19)-9; ch[i]=(i%23)-11;
        h_a[i]=pack(al[i],ah[i]); h_b[i]=pack(bl[i],bh[i]); h_c[i]=pack(cl[i],ch[i]); }
    uint32_t *d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t exp=pack(mn3(al[i],bl[i],cl[i]), mn3(ah[i],bh[i],ch[i]));
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%08x out[%d]=%08x\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);free(al);free(ah);free(bl);free(bh);free(cl);free(ch);
    return pass?0:1;
}
