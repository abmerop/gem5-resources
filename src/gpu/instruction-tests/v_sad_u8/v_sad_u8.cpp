/*
 * HIP test for V_SAD_U8 (VOP3 sum of absolute byte differences plus src2).
 *
 * Verified against gem5's exact expression in
 * src/arch/amdgpu/common/insts/vop3.cc.  Inputs are exact integers.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_sad_u8 v_sad_u8.cpp
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

static const int N = 256;
static const int BLOCK_SIZE = 64;

__global__ void
kernel_v_sad_u8(const uint32_t *a, const uint32_t *b, const uint32_t *c,
              uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t va=a[i], vb=b[i], vc=c[i], vd;
    __asm__ volatile("v_sad_u8 %0, %1, %2, %3\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb), "v"(vc));
    out[i]=vd;
}

static uint32_t ref(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t s=c;
    for (int k=0;k<4;k++){ int ab=(a>>(8*k))&0xff; int bb=(b>>(8*k))&0xff; s += (uint32_t)(ab-bb); }
    return s;
}

int
main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t *h_c=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++) { { uint32_t ba=0,bb=0; for(int k=0;k<4;k++){ int lo=(i*(k+1))%128; int hi=lo+((i+k)%127); ba|=((uint32_t)(hi&0xff))<<(8*k); bb|=((uint32_t)(lo&0xff))<<(8*k);} h_a[i]=ba; h_b[i]=bb; h_c[i]=(uint32_t)(i*3u); } }
    uint32_t *d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4)); GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_c,N*4)); GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel_v_sad_u8,dim3(N/BLOCK_SIZE),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++) {
        uint32_t expected=ref(h_a[i],h_b[i],h_c[i]);
        if (h_out[i]!=expected) {
            fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],expected);
            pass=0; break;
        }
    }
    if (pass) printf("out[0]=0x%08x out[255]=0x%08x\n",h_out[0],h_out[255]);
    printf("%s\n", pass?"PASSED":"FAILED");
    return pass?0:1;
}
