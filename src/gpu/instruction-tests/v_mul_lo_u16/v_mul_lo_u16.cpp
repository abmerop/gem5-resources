/*
 * HIP test for V_MUL_LO_U16 (VOP2 16-bit multiply, low 16 bits).
 * gem5: out = a * b.  Exact integers; base (e32) encoding only (SDWA/DPP hit a
 * gem5 panic_if guard for this op).
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_mul_lo_u16 v_mul_lo_u16.cpp
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
kernel_v_mul_lo_u16(const uint16_t *a, const uint16_t *b, uint16_t *out, int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    uint32_t va=(uint32_t)a[i], vb=(uint32_t)b[i], vd;
    __asm__ volatile("v_mul_lo_u16_e32 %0, %1, %2\n\t" : "=v"(vd) : "v"(va), "v"(vb));
    out[i]=(uint16_t)vd;
}
int main(void) {
    uint16_t *h_a=(uint16_t*)malloc(N*sizeof(uint16_t)),*h_b=(uint16_t*)malloc(N*sizeof(uint16_t)),*h_out=(uint16_t*)malloc(N*sizeof(uint16_t));
    for (int i=0;i<N;i++) { uint16_t a=(uint16_t)(i*97+5); uint16_t b=(uint16_t)(i*53+11); h_a[i]=a; h_b[i]=b; }
    uint16_t *d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(uint16_t))); GPU_CHECK(hipMalloc(&d_b,N*sizeof(uint16_t))); GPU_CHECK(hipMalloc(&d_out,N*sizeof(uint16_t)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(uint16_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(uint16_t),hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel_v_mul_lo_u16,dim3(N/BLOCK_SIZE),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(uint16_t),hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++) {
        uint16_t a=h_a[i], b=h_b[i];
        uint16_t expected=(uint16_t)(a * b);
        if (h_out[i]!=expected) { fprintf(stderr,"FAIL[%d] got %lld exp %lld\n",i,(long long)h_out[i],(long long)expected); pass=0; break; }
    }
    if (pass) printf("out[0]=%lld out[255]=%lld\n",(long long)h_out[0],(long long)h_out[255]);
    printf("%s\n", pass?"PASSED":"FAILED");
    return pass?0:1;
}
