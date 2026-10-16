/*
 * HIP test for V_BITOP3_B16 (VOP3 programmable 3-input bitwise LUT, 16-bit).
 *
 * gem5 (vop3.cc Inst_VOP3__V_BITOP3_B16): reads the low 16 bits of S0/S1/S2
 * (opsel=0), computes per-bit D = LUT(s0,s1,s2) where the 8-bit truth table is
 * packed into NEG[2:0]/ABS[5:3]/OMOD[7:6], and writes the 16-bit result into
 * the low half of VDST (high half preserved).  We use bitop3:0x96 = XOR3, so
 * the reference is (uint16_t)(S0 ^ S1 ^ S2).  Only the low 16 bits are checked.
 * Base only.
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
kern(const uint32_t *a, const uint32_t *b, const uint32_t *c,
     uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t va=a[i], vb=b[i], vc=c[i], vd;
    __asm__ volatile("v_bitop3_b16 %0, %1, %2, %3 bitop3:0x96\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb), "v"(vc));
    out[i] = (uint16_t)vd;
}

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_c=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++){
        h_a[i]=(uint32_t)(i*2654435761u + 0x1111u);
        h_b[i]=(uint32_t)(i*40503u      + 0xabcdu);
        h_c[i]=(uint32_t)(i*0x9e3779b1u + 0x7fffu);
    }
    uint32_t *d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint16_t exp=(uint16_t)(h_a[i]^h_b[i]^h_c[i]);
        if((uint16_t)h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%04x exp 0x%04x\n",
            i,(uint16_t)h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=0x%04x out[%d]=0x%04x\n",(uint16_t)h_out[1],N-1,(uint16_t)h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
