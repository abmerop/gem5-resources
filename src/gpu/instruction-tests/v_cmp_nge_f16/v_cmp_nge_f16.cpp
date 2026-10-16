/*
 * HIP test for V_CMP_NGE_F16 (ENC_VOP3/VOPC, "not greater-equal" f16 compare).
 * gem5 semantics: sdst[lane] = !(S0.f16 >= S1.f16) ? 1 : 0  (true also if NaN).
 * Inline asm forces the VOP3 _e64 encoding; the wave mask goes to an SGPR pair
 * and each lane extracts its own bit.  Pattern A (vector compare).
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
    uint16_t s0=a[i], s1=b[i];
    uint64_t mask;
    __asm__ volatile("v_cmp_nge_f16_e64 %0, %1, %2\n\t"
                     : "=s"(mask) : "v"(s0), "v"(s1));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

static uint16_t hb(float v){ __half h=__float2half(v); uint16_t b; __builtin_memcpy(&b,&h,2); return b; }
static float hf(uint16_t b){ __half h; __builtin_memcpy(&h,&b,2); return __half2float(h); }

int main(void)
{
    uint16_t *h_a=(uint16_t*)malloc(N*2),*h_b=(uint16_t*)malloc(N*2);
    int *h_out=(int*)malloc(N*sizeof(int));
    const uint16_t f16_nan=0x7E00;
    for(int i=0;i<N;i++){
        switch(i%4){
        case 0: h_a[i]=hb((float)(i%9)); h_b[i]=hb((float)(i%5)); break;
        case 1: h_a[i]=hb((float)(i%5)); h_b[i]=hb((float)(i%5)+1.0f); break; /* lt -> !(>=)=true */
        case 2: h_a[i]=f16_nan;          h_b[i]=hb(1.0f); break; /* NaN -> true */
        case 3: h_a[i]=hb(3.0f);         h_b[i]=hb(3.0f); break; /* eq -> >= -> false */
        }
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
        float af=hf(h_a[i]), bf=hf(h_b[i]);
        int exp=!(af>=bf)?1:0;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%d out[2]=%d out[%d]=%d\n",h_out[1],h_out[2],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);
    return pass?0:1;
}
