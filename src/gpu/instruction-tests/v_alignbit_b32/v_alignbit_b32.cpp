/*
 * HIP test for V_ALIGNBIT_B32 (ENC_VOP3).
 * D.u32 = ({S0,S1} >> S2[4:0])[31:0] funnel shift
 * gem5: src/arch/amdgpu/common/insts/vop3.cc. Pattern A (vector): per-lane
 * operands via "v" constraints force the VOP3 encoding. Exact 32-bit check.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                   \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const uint32_t *a, const uint32_t *b, const uint32_t *c,
     uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s0 = a[i], s1 = b[i], s2 = c[i], d;
    __asm__ volatile("v_alignbit_b32 %0, %1, %2, %3\n\t" : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    out[i] = d;
}

static uint32_t ref(uint32_t a, uint32_t b, uint32_t c) { return ((uint32_t)((((uint64_t)a<<32)|b) >> (c & 31))); }

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_c=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++){
        h_a[i]=0x01020304u*(uint32_t)(i+1) ^ (uint32_t)(i<<3);
        h_b[i]=0xdeadbeefu ^ (uint32_t)(i*2654435761u);
        h_c[i]=(uint32_t)(i*0x9e3779b1u)+7u;
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
        uint32_t exp=ref(h_a[i],h_b[i],h_c[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%08x out[%d]=%08x\n",h_out[1],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
