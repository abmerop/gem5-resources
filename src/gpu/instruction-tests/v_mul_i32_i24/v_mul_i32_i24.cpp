/*
 * HIP test for V_MUL_I32_I24 (VOP2 signed 24-bit multiply, low 32 bits of
 * product), with DPP and SDWA variants.
 *
 * D.i32 = sext24(S0) * sext24(S1)   (signed 24-bit operands, 32-bit result)
 *
 * gem5 implements this family via vop2Helper, so SDWA src/dst selects and the
 * DPP lane swizzle are applied to src0 before the op.  OP(s0,s1) =
 * sext24(s0) * sext24(s1) truncated to 32 bits.
 *
 *   base : v_mul_i32_i24_e32 vd, va, vb            out = OP(a[i], b[i])
 *   dpp  : v_mul_i32_i24_dpp ... quad_perm:[1,0,3,2]
 *            src0 gathered from quadBase+perm[lane&3], perm={1,0,3,2}
 *            out = OP(a[swizzled_lane], b[i])
 *   sdwa : v_mul_i32_i24_sdwa ... src0_sel:WORD_1 src1_sel:DWORD dst_sel:DWORD
 *            src0 read from the HIGH 16 bits of a packed register (= a[i]),
 *            out = OP(a[i], b[i]).  UNUSED_PAD (gem5 only allows PRESERVE with
 *            dst_sel:DWORD; PAD is safe here).
 *
 * a[] values are kept to 16 bits (positive) so the SDWA WORD_1 extraction is
 * exact; b[] spans the full signed 24-bit range so signedness is exercised.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_mul_i32_i24 v_mul_i32_i24.cpp
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
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N=256; static const int BLOCK_SIZE=64;

__global__ void
kbase(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t s0=a[i], s1=b[i], vd;
    __asm__ volatile("v_mul_i32_i24_e32 %0, %1, %2\n\t" : "=v"(vd) : "v"(s0), "v"(s1));
    out[i]=vd;
}
__global__ void
kdpp(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t s0=a[i], s1=b[i], vd;
    __asm__ volatile("v_mul_i32_i24_dpp %0, %1, %2 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t"
                     : "=v"(vd) : "v"(s0), "v"(s1));
    out[i]=vd;
}
__global__ void
ksdwa(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t packed=(a[i]&0xffffu)<<16, s1=b[i], vd;
    __asm__ volatile("v_mul_i32_i24_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD "
                     "src0_sel:WORD_1 src1_sel:DWORD\n\t"
                     : "=v"(vd) : "v"(packed), "v"(s1));
    out[i]=vd;
}

static int32_t sext24(uint32_t v) {
    v &= 0xffffffu;
    if (v & 0x800000u) v |= 0xff000000u;
    return (int32_t)v;
}
static uint32_t OP(uint32_t s0, uint32_t s1) {
    int64_t p = (int64_t)sext24(s0) * (int64_t)sext24(s1);
    return (uint32_t)(int32_t)p;
}

int main(void) {
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_base=(uint32_t*)malloc(N*4),*h_dpp=(uint32_t*)malloc(N*4),*h_sdwa=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++) { h_a[i]=(uint32_t)((i*101+7) & 0xffff); h_b[i]=(uint32_t)((i*131+5) & 0xffffff); }
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
        uint32_t eb=OP(h_a[i],h_b[i]);
        if (h_base[i]!=eb) { fprintf(stderr,"FAIL base[%d] got 0x%08x exp 0x%08x\n",i,h_base[i],eb); pass=0; break; }
        int lane=i%BLOCK_SIZE, wb=i-lane, sl=(lane&~3)+perm[lane&3];
        uint32_t ed=OP(h_a[wb+sl],h_b[i]);
        if (h_dpp[i]!=ed) { fprintf(stderr,"FAIL dpp[%d] got 0x%08x exp 0x%08x\n",i,h_dpp[i],ed); pass=0; break; }
        uint32_t es=OP(h_a[i],h_b[i]);
        if (h_sdwa[i]!=es) { fprintf(stderr,"FAIL sdwa[%d] got 0x%08x exp 0x%08x\n",i,h_sdwa[i],es); pass=0; break; }
    }
    if (pass) printf("base[1]=0x%08x dpp[1]=0x%08x sdwa[1]=0x%08x\n",h_base[1],h_dpp[1],h_sdwa[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
