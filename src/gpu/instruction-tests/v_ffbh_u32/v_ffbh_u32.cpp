/*
 * HIP test for V_FFBH_U32 (VOP1 find-first-bit-high, unsigned).
 *
 * V_FFBH_U32 returns the number of leading zero bits, i.e. the count of 0s
 * from the MSB down to the first set bit (0..31), or 0xffffffff if src==0.
 * (This is clz, NOT the bit index of the MSB.)  SDWA/DPP hit a panic_if guard
 * in gem5, so only the base encoding is exercised (variants stay not-working).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_ffbh_u32 v_ffbh_u32.cpp
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
kernel_v_ffbh_u32(const uint32_t *a, uint32_t *out, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    uint32_t va=a[i], vd;
    __asm__ volatile("v_ffbh_u32_e32 %0, %1\n\t" : "=v"(vd) : "v"(va));
    out[i]=vd;
}

int main(void) {
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++) h_a[i] = (i==0)?0u : (uint32_t)(0x80000000u >> (i%32)) | (uint32_t)(i*131u);
    uint32_t *d_a,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4)); GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel_v_ffbh_u32,dim3(N/BLOCK_SIZE),dim3(BLOCK_SIZE),0,0,d_a,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++) {
        uint32_t v=h_a[i];
        uint32_t expected = (v==0)?0xffffffffu : (uint32_t)__builtin_clz(v);
        if (h_out[i]!=expected) { fprintf(stderr,"FAIL[%d] in=0x%08x got %u exp %u\n",i,v,h_out[i],expected); pass=0; break; }
    }
    if (pass) printf("out[0]=%u out[1]=%u\n",h_out[0],h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
