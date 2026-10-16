/*
 * HIP test for S_WAKEUP (ENC_SOPP, wake up other waves in the workgroup).
 *
 * The instruction has no data effect observable from a single-wave kernel; the
 * test simply emits it and verifies a subsequent computation still completes
 * correctly (i.e. the instruction decoded and executed without aborting the
 * simulation).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_wakeup s_wakeup.cpp
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
kernel(const uint32_t *a, const uint32_t *b, uint32_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    __asm__ volatile("s_wakeup\n\t" ::: "memory");
    out[i] = a[i] + b[i];
}

int main(void) {
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++){ h_a[i]=(uint32_t)(i*2654435761u+13); h_b[i]=(uint32_t)(0x9e3779b9u*(i+1)); }
    uint32_t *d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4)); GPU_CHECK(hipMalloc(&d_b,N*4)); GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    int g=N/BLOCK_SIZE;
    hipLaunchKernelGGL(kernel,dim3(g),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_out,N); GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++){ uint32_t e=h_a[i]+h_b[i]; if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],e);pass=0;break;} }
    if(pass) printf("out[1]=0x%08x out[2]=0x%08x\n",h_out[1],h_out[2]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
