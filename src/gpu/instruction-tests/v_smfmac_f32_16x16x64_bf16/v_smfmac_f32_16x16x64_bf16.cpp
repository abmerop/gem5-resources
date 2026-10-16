/*
 * HIP test for v_smfmac_f32_16x16x64_bf16 (VOP3P-MAI sparse matrix, bf16 inputs, f32 acc,
 * gfx950/CDNA4).  Pattern D.
 *
 *   D.f32 = A_sparse.bf16 * B.bf16 + C.f32   (4:2 sparsity, K=64)
 *
 * Operand layout (llvm-mc -mcpu=gfx950): A=v[0:3] (4 VGPRs, 8 bf16/lane),
 * B=v[4:11] (8 VGPRs, 16 bf16/lane), idx=v12, C/D=a[0:3] (4 f32/lane).
 * Every A,B element = 1.0 (bf16 1.0 = 0x3F80), C=0.  Sparse MFMA sums
 * K/2 = 32 products of 1.0*1.0; 32.0 is exact in f32 -> every output == 32.0.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_smfmac_f32_16x16x64_bf16 v_smfmac_f32_16x16x64_bf16.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

static const int ADW = 4;
static const int BDW = 8;
static const int CD  = 4;
static const uint32_t ONE_x2 = 0x3F803F80u;

typedef __attribute__((ext_vector_type(4))) int int4_v;
typedef __attribute__((ext_vector_type(8))) int int8_v;
typedef __attribute__((ext_vector_type(4))) float float4_v;

__global__ void
mfma_kernel(const uint32_t *a_in, const uint32_t *b_in,
            const float *c_in, float *d_out, int num_waves)
{
    int wf_id = blockIdx.x, lane = threadIdx.x;
    if (wf_id >= num_waves) return;
    int base = wf_id * 64 + lane;
    int4_v va;  for (int i=0;i<ADW;i++) va[i]=(int)a_in[base*ADW+i];
    int8_v vb;  for (int i=0;i<BDW;i++) vb[i]=(int)b_in[base*BDW+i];
    float4_v vd; for (int i=0;i<CD;i++) vd[i]=c_in[base*CD+i];
    uint32_t idx = 0x44444444u;
    __asm__ volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "v_smfmac_f32_16x16x64_bf16 %0, %1, %2, %3\n\t"
        : "+a"(vd) : "v"(va), "v"(vb), "v"(idx));
    for (int i=0;i<CD;i++) d_out[base*CD+i]=vd[i];
}

int
main(void)
{
    const int num_waves = 4, threads = num_waves * 64;
    uint32_t *h_a=(uint32_t*)malloc((size_t)threads*ADW*4);
    uint32_t *h_b=(uint32_t*)malloc((size_t)threads*BDW*4);
    float *h_c=(float*)malloc((size_t)threads*CD*4);
    float *h_d=(float*)malloc((size_t)threads*CD*4);
    for (int t=0;t<threads;t++) {
        for (int i=0;i<ADW;i++) h_a[t*ADW+i]=ONE_x2;
        for (int i=0;i<BDW;i++) h_b[t*BDW+i]=ONE_x2;
        for (int i=0;i<CD;i++)  h_c[t*CD+i]=0;
    }
    uint32_t *d_a,*d_b; float *d_c,*d_d;
    GPU_CHECK(hipMalloc(&d_a,(size_t)threads*ADW*4));
    GPU_CHECK(hipMalloc(&d_b,(size_t)threads*BDW*4));
    GPU_CHECK(hipMalloc(&d_c,(size_t)threads*CD*4));
    GPU_CHECK(hipMalloc(&d_d,(size_t)threads*CD*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,(size_t)threads*ADW*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,(size_t)threads*BDW*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,(size_t)threads*CD*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d,0,(size_t)threads*CD*4));
    hipLaunchKernelGGL(mfma_kernel, dim3(num_waves), dim3(64), 0, 0, d_a,d_b,d_c,d_d,num_waves);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_d,d_d,(size_t)threads*CD*4,hipMemcpyDeviceToHost));
    const float EXPECTED = 32.0f;
    int pass=1, checked=0;
    for (int t=0;t<threads && pass;t++)
        for (int i=0;i<CD;i++) {
            float got = h_d[t*CD+i];
            if (fabsf(got - EXPECTED) > 0.01f) {
                fprintf(stderr,"FAIL t=%d [%d]: got %.3f expected %.3f\n",t,i,got,EXPECTED);
                pass=0; break;
            }
            checked++;
        }
    printf("v_smfmac_f32_16x16x64_bf16: checked %d, expected %.3f\n", checked, EXPECTED);
    if (pass) printf("  D[0]=%.1f D[last]=%.1f\n", h_d[0], h_d[threads*CD-1]);
    printf("%s\n", pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_d));
    free(h_a);free(h_b);free(h_c);free(h_d);
    return pass?0:1;
}
