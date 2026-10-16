/*
 * HIP test for S_SETREG_B32 (ENC_SOPK, write an SGPR value into a field of a
 * hardware register).  Pattern B.
 *
 * Per gem5 src/arch/amdgpu/common/insts/sopk.cc, S_SETREG_B32 reads SDST (the
 * source SGPR), and only models writes to HW_REG_MODE with size==2 and offset
 * 0 or 4 (the FP round/denorm mode fields) which it treats as a no-op; every
 * other target hits panicUnimplemented.  So this test writes the FP round-mode
 * field of the MODE register from an SGPR value (a supported, side-effect-free
 * form) and then verifies that a simple computation still completes correctly
 * -- i.e. the instruction decoded and executed without panicking.
 *
 *   s_setreg_b32 hwreg(HW_REG_MODE, 0, 2), <sgpr>
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_setreg_b32 s_setreg_b32.cpp
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

static const int N = 256; static const int BLOCK_SIZE = 64;

__global__ void
kernel(const uint32_t *a, const uint32_t *b, uint32_t *out, uint32_t mode,
       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    /* Set the FP round-mode field of MODE from an SGPR (supported no-op). */
    __asm__ volatile("s_setreg_b32 hwreg(HW_REG_MODE, 0, 2), %0\n\t"
                     :: "s"(mode) : "memory");
    out[i] = a[i] + b[i];
}

int main(void) {
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++){ h_a[i]=(uint32_t)(i*2654435761u+13); h_b[i]=(uint32_t)(0x9e3779b9u*(i+1)); }
    uint32_t *d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4)); GPU_CHECK(hipMalloc(&d_b,N*4)); GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    int g=N/BLOCK_SIZE;
    hipLaunchKernelGGL(kernel,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_out,0u,N); GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++){ uint32_t e=h_a[i]+h_b[i]; if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],e);pass=0;break;} }
    if(pass) printf("out[1]=0x%08x out[2]=0x%08x\n",h_out[1],h_out[2]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
