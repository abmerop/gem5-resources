/*
 * HIP test for V_MED3_F16 (ENC_VOP3, median of three f16).
 * gem5 semantics: D.f16 = median(S0.f16, S1.f16, S2.f16).
 * Inline asm forces the VOP3 encoding; per-lane f16 operands.
 * Distinct integer-valued f16 inputs -> exact, bit-exact compare valid.
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
kern(const uint16_t *a, const uint16_t *b, const uint16_t *c, uint16_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t s0=a[i], s1=b[i], s2=c[i], d;
    __asm__ volatile("v_med3_f16 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

static uint16_t hb(int v){ __half h=__float2half((float)v); uint16_t b; __builtin_memcpy(&b,&h,2); return b; }
static int med3(int x,int y,int z){ int mn=x<y?x:y, mx=x<y?y:x; if(z<mn)return mn; if(z>mx)return mx; return z; }

int main(void)
{
    uint16_t *h_a=(uint16_t*)malloc(N*2),*h_b=(uint16_t*)malloc(N*2),*h_c=(uint16_t*)malloc(N*2),*h_out=(uint16_t*)malloc(N*2);
    int *va=(int*)malloc(N*4),*vb=(int*)malloc(N*4),*vc=(int*)malloc(N*4);
    for(int i=0;i<N;i++){ va[i]=(i*7)%101-50; vb[i]=(i*13)%97-48; vc[i]=(i*5)%89-44;
        h_a[i]=hb(va[i]); h_b[i]=hb(vb[i]); h_c[i]=hb(vc[i]); }
    uint16_t *d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));GPU_CHECK(hipMalloc(&d_c,N*2));GPU_CHECK(hipMalloc(&d_out,N*2));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*2,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*2,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint16_t exp=hb(med3(va[i],vb[i],vc[i]));
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %04x exp %04x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%04x out[%d]=%04x\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);free(va);free(vb);free(vc);
    return pass?0:1;
}
