/*
 * HIP test for V_PK_MIN_I16 (ENC_VOP3P, packed signed 16-bit minimum).
 *
 *   D[15:0]  = min(S0[15:0],  S1[15:0])    (signed)
 *   D[31:16] = min(S0[31:16], S1[31:16])   (signed)
 *
 * Pattern A (vector), base only.  Matches gem5 vop3p.cc opImpl (signed).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_pk_min_i16 v_pk_min_i16.cpp
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
kernel(const uint32_t *a, const uint32_t *b, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t va = a[i], vb = b[i], vd;
    __asm__ volatile("v_pk_min_i16 %0, %1, %2\n\t" : "=v"(vd) : "v"(va), "v"(vb));
    out[i] = vd;
}

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){
        int16_t a0=(int16_t)(i*131-9000), a1=(int16_t)(-i*177+5000);
        int16_t b0=(int16_t)(-i*53+3000), b1=(int16_t)(i*701-12000);
        h_a[i]=((uint32_t)(uint16_t)a1<<16)|(uint16_t)a0;
        h_b[i]=((uint32_t)(uint16_t)b1<<16)|(uint16_t)b0;
    }
    uint32_t *d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int16_t a0=h_a[i]&0xffff,a1=h_a[i]>>16,b0=h_b[i]&0xffff,b1=h_b[i]>>16;
        int16_t lo=a0<b0?a0:b0, hi=a1<b1?a1:b1;
        uint32_t exp=((uint32_t)(uint16_t)hi<<16)|(uint16_t)lo;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
