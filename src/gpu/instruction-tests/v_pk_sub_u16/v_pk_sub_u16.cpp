/*
 * HIP test for V_PK_SUB_U16 (ENC_VOP3P, packed u16x2 subtract).
 * gem5 semantics: D[15:0]=S0[15:0]-S1[15:0], D[31:16]=S0[31:16]-S1[31:16].
 * Inline asm forces the VOP3P encoding; each operand is a packed u16x2 dword.
 * 16-bit modular arithmetic -> exact compare.
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
kern(const uint32_t *a, const uint32_t *b, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s0=a[i], s1=b[i], d;
    __asm__ volatile("v_pk_sub_u16 %0, %1, %2\n\t" : "=v"(d) : "v"(s0), "v"(s1));
    out[i] = d;
}

static uint32_t pk(uint16_t lo,uint16_t hi){ return (uint32_t)lo|((uint32_t)hi<<16); }

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    uint16_t *al=(uint16_t*)malloc(N*2),*ah=(uint16_t*)malloc(N*2),*bl=(uint16_t*)malloc(N*2),*bh=(uint16_t*)malloc(N*2);
    for(int i=0;i<N;i++){ al[i]=(uint16_t)(i*131+5); ah[i]=(uint16_t)(i*257+9);
        bl[i]=(uint16_t)(i*97+50000); bh[i]=(uint16_t)(i*61+12345);
        h_a[i]=pk(al[i],ah[i]); h_b[i]=pk(bl[i],bh[i]); }
    uint32_t *d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t exp=pk((uint16_t)(al[i]-bl[i]),(uint16_t)(ah[i]-bh[i]));
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%08x out[%d]=%08x\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);free(al);free(ah);free(bl);free(bh);
    return pass?0:1;
}
