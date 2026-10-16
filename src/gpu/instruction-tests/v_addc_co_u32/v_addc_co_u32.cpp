/*
 * HIP test for V_ADDC_CO_U32 (VOP2 add with carry-in/carry-out via VCC).
 *
 * gem5 routes this through vop2bHelper, which panics on the DPP form
 * ("DPP not implemented for v_addc_co_u32"), so only the base e32 form is
 * exercised.  Carry-in is cleared (vcc=0) so:
 *
 *   out = a + b   (mod 2^32)
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_addc_co_u32 v_addc_co_u32.cpp
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
    uint32_t va=a[i], vb=b[i], vd;
    __asm__ volatile("s_mov_b64 vcc, 0\n\t"
                     "v_addc_co_u32_e32 %0, vcc, %1, %2, vcc\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb) : "vcc");
    out[i]=vd;
}

int main(void) {
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_base=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++){ h_a[i]=(uint32_t)(i*2654435761u+13); h_b[i]=(uint32_t)(0x9e3779b9u*(i+1)); }
    uint32_t *d_a,*d_b,*d_base;
    GPU_CHECK(hipMalloc(&d_a,N*4)); GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_base,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    int g=N/BLOCK_SIZE;
    hipLaunchKernelGGL(kbase,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_base,N); GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base,d_base,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++) {
        uint32_t eb=h_a[i]+h_b[i];
        if (h_base[i]!=eb) { fprintf(stderr,"FAIL base[%d] got 0x%08x exp 0x%08x\n",i,h_base[i],eb); pass=0; break; }
    }
    if (pass) printf("base[1]=0x%08x base[2]=0x%08x\n",h_base[1],h_base[2]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
