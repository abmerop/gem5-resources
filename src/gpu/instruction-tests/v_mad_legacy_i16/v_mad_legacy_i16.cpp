/*
 * HIP test for V_MAD_LEGACY_I16 (ENC_VOP3, 16-bit signed multiply-add).
 *   D.i16 = S0.i16 * S1.i16 + S2.i16   (low 16 bits, wraps mod 2^16)
 *
 * gem5 decodes v_mad_legacy_i16 to Inst_VOP3__V_MAD_I16 (identical integer
 * semantics).  Explicit inline asm forces the VOP3 encoding.  Pattern A
 * (per-lane "v" operands).
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o v_mad_legacy_i16 v_mad_legacy_i16.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

static const int N=256,BLOCK_SIZE=64;

__global__ void
kern(const int16_t *a, const int16_t *b, const int16_t *c, int16_t *out, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* load into 32-bit regs for the "v" constraint, values are i16 */
    int s0 = a[i], s1 = b[i], s2 = c[i];
    int d;
    __asm__ volatile("v_mad_legacy_i16 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = (int16_t)d;
}

int
main(void)
{
    int16_t *h_a=(int16_t*)malloc(N*2),*h_b=(int16_t*)malloc(N*2);
    int16_t *h_c=(int16_t*)malloc(N*2),*h_out=(int16_t*)malloc(N*2);
    for (int i=0;i<N;i++){
        h_a[i]=(int16_t)(i*37-500);
        h_b[i]=(int16_t)(-i*13+200);
        h_c[i]=(int16_t)(i*7-1000);
    }
    int16_t *d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));
    GPU_CHECK(hipMalloc(&d_c,N*2));GPU_CHECK(hipMalloc(&d_out,N*2));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*2,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*2,hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++){
        int16_t exp=(int16_t)((uint16_t)(h_a[i]*h_b[i]+h_c[i]));
        if (h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",
            i,h_out[i],exp);pass=0;break;}
    }
    if (pass) printf("out[1]=%d out[%d]=%d\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));
    GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
