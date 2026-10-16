/*
 * HIP test for V_LDEXP_F16 (VOP2 f16 ldexp: D = S0.f16 * 2^S1.i), with DPP and
 * SDWA variants.
 *
 *   D.f16 = S0.f16 * 2^(S1.i)
 *
 * Inputs are chosen so every result is exactly representable in f16 (S0 is a
 * small exact value, S1 a small exponent), so verification uses exact bit
 * comparison after host ldexpf + round-to-f16.  gem5 routes the DPP/SDWA forms
 * through vop2Helper, which applies the swizzle/select to src0:
 *
 *   base : v_ldexp_f16_e32 vd, va, vb         vd = ldexp(a[i], b[i])
 *   dpp  : v_ldexp_f16_dpp ... quad_perm:[1,0,3,2]  vd = ldexp(a[swizzled], b[i])
 *   sdwa : v_ldexp_f16_sdwa ... src0_sel:WORD_0 src1_sel:DWORD  vd = ldexp(a[i], b[i])
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_ldexp_f16 v_ldexp_f16.cpp
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N=256; static const int BLOCK_SIZE=64;

__global__ void
kbase(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t va=a[i], vb=b[i], vd;
    __asm__ volatile("v_ldexp_f16_e32 %0, %1, %2\n\t" : "=v"(vd) : "v"(va), "v"(vb));
    out[i]=vd & 0xffffu;
}
__global__ void
kdpp(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t va=a[i], vb=b[i], vd;
    __asm__ volatile("v_ldexp_f16_dpp %0, %1, %2 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb));
    out[i]=vd & 0xffffu;
}
__global__ void
ksdwa(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t va=a[i], vb=b[i], vd;
    __asm__ volatile("v_ldexp_f16_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD "
                     "src0_sel:WORD_0 src1_sel:DWORD\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb));
    out[i]=vd & 0xffffu;
}

/* host reference: ldexp the f16 value (exact), round back to f16 bits */
static uint16_t ref(uint16_t a_bits, int e) {
    __half h; __builtin_memcpy(&h, &a_bits, 2);
    float f = __half2float(h);
    float r = ldexpf(f, e);
    __half hr = __float2half(r);
    uint16_t out; __builtin_memcpy(&out, &hr, 2);
    return out;
}

int main(void) {
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_base=(uint32_t*)malloc(N*4),*h_dpp=(uint32_t*)malloc(N*4),*h_sdwa=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++) {
        /* exact f16 mantissa values: {1.0,1.5,2.0,1.25,...}, exponents -2..3 */
        float base_vals[8]={1.0f,1.5f,2.0f,1.25f,0.5f,0.75f,1.75f,1.0f};
        __half h=__float2half(base_vals[i&7]);
        uint16_t hb; __builtin_memcpy(&hb,&h,2);
        h_a[i]=hb;
        h_b[i]=(uint32_t)((i%6)-2);   /* exponent -2..3 */
    }
    uint32_t *d_a,*d_b,*d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,N*4)); GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_base,N*4)); GPU_CHECK(hipMalloc(&d_dpp,N*4)); GPU_CHECK(hipMalloc(&d_sdwa,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    int g=N/BLOCK_SIZE;
    hipLaunchKernelGGL(kbase,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_base,N); GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kdpp,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_dpp,N); GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(ksdwa,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_sdwa,N); GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base,d_base,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp,d_dpp,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa,d_sdwa,N*4,hipMemcpyDeviceToHost));
    static const int perm[4]={1,0,3,2};
    int pass=1;
    for (int i=0;i<N;i++) {
        uint16_t eb=ref((uint16_t)h_a[i],(int)h_b[i]);
        if ((uint16_t)h_base[i]!=eb) { fprintf(stderr,"FAIL base[%d] got 0x%04x exp 0x%04x\n",i,h_base[i],eb); pass=0; break; }
        int lane=i%BLOCK_SIZE, wb=i-lane, sl=(lane&~3)+perm[lane&3];
        uint16_t ed=ref((uint16_t)h_a[wb+sl],(int)h_b[i]);
        if ((uint16_t)h_dpp[i]!=ed) { fprintf(stderr,"FAIL dpp[%d] got 0x%04x exp 0x%04x\n",i,h_dpp[i],ed); pass=0; break; }
        uint16_t es=ref((uint16_t)h_a[i],(int)h_b[i]);
        if ((uint16_t)h_sdwa[i]!=es) { fprintf(stderr,"FAIL sdwa[%d] got 0x%04x exp 0x%04x\n",i,h_sdwa[i],es); pass=0; break; }
    }
    if (pass) printf("base[1]=0x%04x dpp[1]=0x%04x sdwa[1]=0x%04x\n",h_base[1],h_dpp[1],h_sdwa[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
