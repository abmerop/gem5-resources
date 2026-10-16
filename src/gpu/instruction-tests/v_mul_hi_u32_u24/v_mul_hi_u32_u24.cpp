/*
 * HIP test for V_MUL_HI_U32_U24 (VOP2 high 32 bits of unsigned 24-bit product).
 * gem5: out = (uint32_t)(((uint64_t)(a & 0xffffffu) * (uint64_t)(b & 0xffffffu)) >> 32).  Exact integers; base (e32) encoding only (SDWA/DPP hit a
 * gem5 panic_if guard for this op).
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_mul_hi_u32_u24 v_mul_hi_u32_u24.cpp
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
kernel_v_mul_hi_u32_u24(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t va=(uint32_t)a[i], vb=(uint32_t)b[i], vd;
    __asm__ volatile("v_mul_hi_u32_u24_e32 %0, %1, %2\n\t" : "=v"(vd) : "v"(va), "v"(vb));
    out[i]=(uint32_t)vd;
}
int main(void) {
    uint32_t *h_a=(uint32_t*)malloc(N*sizeof(uint32_t)),*h_b=(uint32_t*)malloc(N*sizeof(uint32_t)),*h_out=(uint32_t*)malloc(N*sizeof(uint32_t));
    for (int i=0;i<N;i++) { uint32_t a=(uint32_t)(0xffffffu - (uint32_t)i*1237u); uint32_t b=(uint32_t)(0xfffff0u - (uint32_t)i*5051u); h_a[i]=a; h_b[i]=b; }
    uint32_t *d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(uint32_t))); GPU_CHECK(hipMalloc(&d_b,N*sizeof(uint32_t))); GPU_CHECK(hipMalloc(&d_out,N*sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(uint32_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(uint32_t),hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel_v_mul_hi_u32_u24,dim3(N/BLOCK_SIZE),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++) {
        uint32_t a=h_a[i], b=h_b[i];
        uint32_t expected=(uint32_t)((uint32_t)(((uint64_t)(a & 0xffffffu) * (uint64_t)(b & 0xffffffu)) >> 32));
        if (h_out[i]!=expected) { fprintf(stderr,"FAIL[%d] got %lld exp %lld\n",i,(long long)h_out[i],(long long)expected); pass=0; break; }
    }
    if (pass) printf("out[0]=%lld out[255]=%lld\n",(long long)h_out[0],(long long)h_out[255]);
    printf("%s\n", pass?"PASSED":"FAILED");
    return pass?0:1;
}
