/*
 * HIP test for V_LSHLREV_B16 (VOP2 16-bit logical left shift, reversed operands), with DPP and SDWA variants.
 *
 * gem5 implements this via vop2Helper (no panic guard), so the SDWA src/dst
 * selects and DPP lane swizzle are applied to src0 before the op.  Result per
 * out = OP(src0, src1) where OP(s0,s1) = (s1 << (s0 & 15u)) & 0xffff
 * (16-bit result; upper 16 bits of the destination are zero).
 *
 *   base : v_lshlrev_b16_e32 vd, va, vb           out = OP(a[i], b[i])
 *   dpp  : v_lshlrev_b16_dpp ... quad_perm:[1,0,3,2]
 *            On hardware the DPP lane swizzle is applied to the data operand
 *            (the 2nd source, b), not the shift count (a).  So b is gathered
 *            from quadBase+perm[lane&3], perm={1,0,3,2}.
 *            out = OP(a[i], b[swizzled_lane])
 *   sdwa : v_lshlrev_b16_sdwa ... src0_sel:WORD_1 src1_sel:DWORD dst_sel:DWORD
 *            src0 read from the HIGH 16 bits of a packed register (= a[i]),
 *            out = OP(a[i], b[i]).  UNUSED_PAD (gem5 only allows PRESERVE with
 *            dst_sel:DWORD; PAD is safe here).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_lshlrev_b16 v_lshlrev_b16.cpp
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
    __asm__ volatile("v_lshlrev_b16_e32 %0, %1, %2\n\t" : "=v"(vd) : "v"(s0), "v"(s1));
    out[i]=vd;
}
__global__ void
kdpp(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t s0=a[i], s1=b[i], vd;
    __asm__ volatile("v_lshlrev_b16_dpp %0, %1, %2 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t"
                     : "=v"(vd) : "v"(s0), "v"(s1));
    out[i]=vd;
}
__global__ void
ksdwa(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t packed=(a[i]&0xffffu)<<16, s1=b[i], vd;
    __asm__ volatile("v_lshlrev_b16_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD "
                     "src0_sel:WORD_1 src1_sel:DWORD\n\t"
                     : "=v"(vd) : "v"(packed), "v"(s1));
    out[i]=vd;
}

/* V_LSHLREV_B16 is a 16-bit op: the result is the low 16 bits of the shift
 * and the upper 16 bits of the destination are zero. */
static uint32_t OP(uint32_t s0, uint32_t s1) { return ((s1 << (s0 & 15u)) & 0xffffu); }

int main(void) {
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_base=(uint32_t*)malloc(N*4),*h_dpp=(uint32_t*)malloc(N*4),*h_sdwa=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++) { h_a[i]=(uint32_t)(i % 16)&0xffffu; h_b[i]=(uint32_t)((uint32_t)((i*7+3) & 0xffff)); }
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
        uint32_t ed=OP(h_a[i],h_b[wb+sl]);
        if (h_dpp[i]!=ed) { fprintf(stderr,"FAIL dpp[%d] got 0x%08x exp 0x%08x\n",i,h_dpp[i],ed); pass=0; break; }
        uint32_t es=OP(h_a[i],h_b[i]);
        if (h_sdwa[i]!=es) { fprintf(stderr,"FAIL sdwa[%d] got 0x%08x exp 0x%08x\n",i,h_sdwa[i],es); pass=0; break; }
    }
    if (pass) printf("base[1]=0x%08x dpp[1]=0x%08x sdwa[1]=0x%08x\n",h_base[1],h_dpp[1],h_sdwa[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
