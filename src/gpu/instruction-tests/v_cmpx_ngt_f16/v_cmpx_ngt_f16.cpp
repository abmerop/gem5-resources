/*
 * HIP test for V_CMPX_NGT_F16 (v_cmpx_ngt_f16_e64), CDNA4 / gfx950.
 * Self-checking CMPX test with FINITE inputs only (no NaN), so the
 * predicate is unambiguous: result == (av<=bv). Each lane extracts its bit
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

static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else { int e = 127-15+1; while(!(mant&0x400)){mant<<=1;e--;} mant&=0x3FF;
               f = sign | ((uint32_t)e<<23) | (mant<<13); }
    } else if (exp == 0x1F) { f = sign | 0x7F800000u | (mant<<13);
    } else { f = sign | ((exp-15+127)<<23) | (mant<<13); }
    float r; memcpy(&r,&f,4); return r;
}

__global__ void
kernel_test(const uint16_t *a, const uint16_t *b, int *out, int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint32_t va=(uint32_t)a[i], vb=(uint32_t)b[i];
    uint64_t mask, saved;
    __asm__ volatile(
        "s_mov_b64 %1, exec\n\t"
        "v_cmpx_ngt_f16_e64 %0, %2, %3\n\t"
        "s_mov_b64 exec, %1\n\t"
        : "=s"(mask), "=s"(saved) : "v"(va), "v"(vb));
    out[i]=(int)((mask>>threadIdx.x)&1ULL);
}

int main(void)
{
    uint16_t *h_a=(uint16_t*)malloc(N*sizeof(uint16_t));
    uint16_t *h_b=(uint16_t*)malloc(N*sizeof(uint16_t));
    int *h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){
        switch (i % 6) {
        case 0: h_a[i]=0x3E00; h_b[i]=0x3E00; break; /* 1.5==1.5 */
        case 1: h_a[i]=0x3C00; h_b[i]=0x4000; break; /* 1.0<2.0 */
        case 2: h_a[i]=0x4000; h_b[i]=0x3C00; break; /* 2.0>1.0 */
        case 3: h_a[i]=0xC000; h_b[i]=0xBC00; break; /* -2.0<-1.0 */
        case 4: h_a[i]=0x4500; h_b[i]=0x4500; break; /* 5.0==5.0 */
        case 5: h_a[i]=0x0000; h_b[i]=0x8000; break; /* +0.0==-0.0 */
        }
    }
    uint16_t *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(uint16_t)));GPU_CHECK(hipMalloc(&d_b,N*sizeof(uint16_t)));GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(uint16_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(uint16_t),hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel_test,dim3(N/BLOCK_SIZE),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){
        float av=h2f(h_a[i]); float bv=h2f(h_b[i]);
        int exp=((av<=bv))?1:0;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL out[%d] case %d: got %d exp %d\n",i,i%6,h_out[i],exp);pass=0;}
    }
    if(pass)printf("out[0..5]=%d%d%d%d%d%d\n",h_out[0],h_out[1],h_out[2],h_out[3],h_out[4],h_out[5]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out); return pass?0:1;
}
