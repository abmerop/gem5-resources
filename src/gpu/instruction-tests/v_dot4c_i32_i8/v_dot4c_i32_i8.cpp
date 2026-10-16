/*
 * HIP test for V_DOT4C_I32_I8 (VOP2 4-element signed-byte dot product with
 * accumulate into dst), plus the DPP variant.
 *
 * D.i32 = D.i32 + sum_{k=0}^{3} sext8(S0.byte[k]) * sext8(S1.byte[k])
 *
 * The "C" form reads the destination as the accumulator (src2 == dst), so the
 * inline asm uses "+v".  gem5 routes the DPP form through vop2Helper, which
 * applies the lane swizzle to src0 only (src1 and the dst accumulator are this
 * lane's own values):
 *
 *   base : v_dot4c_i32_i8_e32 vd, va, vb        vd = c[i] + dot(a[i], b[i])
 *   dpp  : v_dot4c_i32_i8_dpp ... quad_perm:[1,0,3,2]
 *            vd = c[i] + dot(a[swizzled_lane], b[i])
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_dot4c_i32_i8 v_dot4c_i32_i8.cpp
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
kbase(const uint32_t *a, const uint32_t *b, const int32_t *c, int32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t va=a[i], vb=b[i]; int32_t vd=c[i];
    __asm__ volatile("v_dot4c_i32_i8_e32 %0, %1, %2\n\t" : "+v"(vd) : "v"(va), "v"(vb));
    out[i]=vd;
}
__global__ void
kdpp(const uint32_t *a, const uint32_t *b, const int32_t *c, int32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t va=a[i], vb=b[i]; int32_t vd=c[i];
    __asm__ volatile("v_dot4c_i32_i8_dpp %0, %1, %2 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t"
                     : "+v"(vd) : "v"(va), "v"(vb));
    out[i]=vd;
}

static int32_t ref(uint32_t a, uint32_t b, int32_t c) {
    int32_t s=c;
    for (int k=0;k<4;k++){
        int8_t ab=(int8_t)((a>>(8*k))&0xff), bb=(int8_t)((b>>(8*k))&0xff);
        s += (int32_t)ab * (int32_t)bb;
    }
    return s;
}

int main(void) {
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    int32_t  *h_c=(int32_t*)malloc(N*4),*h_base=(int32_t*)malloc(N*4),*h_dpp=(int32_t*)malloc(N*4);
    for (int i=0;i<N;i++){ h_a[i]=(uint32_t)(i*2654435761u); h_b[i]=(uint32_t)(0x9e3779b9u*(i+1)); h_c[i]=(int32_t)(i*7-100); }
    uint32_t *d_a,*d_b; int32_t *d_c,*d_base,*d_dpp;
    GPU_CHECK(hipMalloc(&d_a,N*4)); GPU_CHECK(hipMalloc(&d_b,N*4)); GPU_CHECK(hipMalloc(&d_c,N*4));
    GPU_CHECK(hipMalloc(&d_base,N*4)); GPU_CHECK(hipMalloc(&d_dpp,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    int g=N/BLOCK_SIZE;
    hipLaunchKernelGGL(kbase,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_c,d_base,N); GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(kdpp,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_c,d_dpp,N); GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base,d_base,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp,d_dpp,N*4,hipMemcpyDeviceToHost));
    static const int perm[4]={1,0,3,2};
    int pass=1;
    for (int i=0;i<N;i++) {
        int32_t eb=ref(h_a[i],h_b[i],h_c[i]);
        if (h_base[i]!=eb) { fprintf(stderr,"FAIL base[%d] got %d exp %d\n",i,h_base[i],eb); pass=0; break; }
        int lane=i%BLOCK_SIZE, wb=i-lane, sl=(lane&~3)+perm[lane&3];
        int32_t ed=ref(h_a[wb+sl],h_b[i],h_c[i]);
        if (h_dpp[i]!=ed) { fprintf(stderr,"FAIL dpp[%d] got %d exp %d\n",i,h_dpp[i],ed); pass=0; break; }
    }
    if (pass) printf("base[1]=%d dpp[1]=%d\n",h_base[1],h_dpp[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
