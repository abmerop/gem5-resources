/*
 * HIP test for v_mfma_i32_32x32x16_i8 (VOP3P-MAI matrix instruction, gfx950/CDNA4).
 *
 * Uniform-input verification trick (mapping-independent): every A element and
 * every B element = 1.0, C = 0.  Then every output element =
 *   K = 16  (MFMA: sum of K unit products).
 * A dtype = i8, B dtype = i8, accumulate into i32.
 * Operand widths (llvm-mc gfx950): A 2 VGPR(s), B 2 VGPR(s), C/D 16 AGPR(s).
 *
 * Build:  hipcc -O2 --offload-arch=gfx950 -o v_mfma_i32_32x32x16_i8 v_mfma_i32_32x32x16_i8.cpp
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                             \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int ADW = 2;
static const int BDW = 2;
static const int CD  = 16;

typedef __attribute__((ext_vector_type(2))) int int2_v;
typedef __attribute__((ext_vector_type(16))) int int16_v;

__global__ void
mfma_kernel(const uint32_t *a_in, const uint32_t *b_in,
            const int *c_in, int *d_out, int num_waves)
{
    int wf_id = blockIdx.x;
    int lane  = threadIdx.x;
    if (wf_id >= num_waves) return;
    int base = wf_id * 64 + lane;

    int2_v va;
    for (int i=0;i<2;i++) va[i]=(int)a_in[base*2+i];
    int2_v vb;
    for (int i=0;i<2;i++) vb[i]=(int)b_in[base*2+i];
    int16_v vd;
    for (int i=0;i<16;i++) vd[i]=c_in[base*16+i];

    __asm__ volatile(
        "s_waitcnt vmcnt(0)\n\t"
        "v_mfma_i32_32x32x16_i8 %0, %1, %2, %0\n\t"
        : "+a"(vd)
        : "v"(va), "v"(vb));

    for (int i=0;i<16;i++) d_out[base*16+i]=vd[i];
}

int
main(void)
{
    const int num_waves = 4;
    const int threads   = num_waves * 64;

    uint32_t *h_a = (uint32_t*)malloc((size_t)threads*ADW*sizeof(uint32_t));
    uint32_t *h_b = (uint32_t*)malloc((size_t)threads*BDW*sizeof(uint32_t));
    int *h_c = (int*)malloc((size_t)threads*CD*sizeof(int));
    int *h_d = (int*)malloc((size_t)threads*CD*sizeof(int));

    for (int t=0;t<threads;t++) {
        for (int i=0;i<ADW;i++) h_a[t*ADW+i]=0x01010101u;
        for (int i=0;i<BDW;i++) h_b[t*BDW+i]=0x01010101u;
        for (int i=0;i<CD;i++)  h_c[t*CD+i]=0;
    }

    uint32_t *d_a,*d_b; int *d_c,*d_d;
    GPU_CHECK(hipMalloc(&d_a,(size_t)threads*ADW*sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_b,(size_t)threads*BDW*sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_c,(size_t)threads*CD*sizeof(int)));
    GPU_CHECK(hipMalloc(&d_d,(size_t)threads*CD*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,(size_t)threads*ADW*sizeof(uint32_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,(size_t)threads*BDW*sizeof(uint32_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,(size_t)threads*CD*sizeof(int),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(d_d,0,(size_t)threads*CD*sizeof(int)));

    hipLaunchKernelGGL(mfma_kernel, dim3(num_waves), dim3(64), 0, 0,
                       d_a,d_b,d_c,d_d,num_waves);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_d,d_d,(size_t)threads*CD*sizeof(int),hipMemcpyDeviceToHost));

    const int EXPECTED = 16;
    int pass=1, checked=0;
    for (int t=0;t<threads && pass;t++) {
        for (int i=0;i<CD;i++) {
            int got = h_d[t*CD+i];
            if (got != EXPECTED) {
                fprintf(stderr,"FAIL t=%d [%d]: got %d expected %d\n",t,i,got,EXPECTED);
                pass=0; break;
            }
            checked++;
        }
    }
    printf("v_mfma_i32_32x32x16_i8: checked %d, expected %d\n", checked, EXPECTED);
    printf("%s\n", pass?"PASSED":"FAILED");

    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));
    GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_d));
    free(h_a);free(h_b);free(h_c);free(h_d);
    return pass?0:1;
}
