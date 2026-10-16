/*
 * HIP test for BUFFER_STORE_FORMAT_XYZ (ENC_MUBUF, typed store of 3 components
 * with format conversion).  Pattern C.
 *
 * Stores three 32-bit uint components per element through a buffer resource
 * descriptor (32_32_32 UINT, identity dst_sel) and reads them back, checking
 * each component.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o buffer_store_format_xyz buffer_store_format_xyz.cpp
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
                    hipGetErrorString(e));                                   \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
static const int N = 256; static const int BLOCK_SIZE = 64;

__global__ void
kernel(const uint32_t *src, uint32_t *buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    u32x3 data;
    data.x = src[i*3+0]; data.y = src[i*3+1]; data.z = src[i*3+2];
    uint64_t base = (uint64_t)buf;
    u32x4 srd;
    srd.x = (uint32_t)(base & 0xffffffffu);
    srd.y = (uint32_t)((base >> 32) & 0xffffu);
    srd.z = 0xffffffffu;
    /* word3: dstSel identity (X=4,Y=5,Z=6,W=7), numFmt=UINT(4),
     * dataFmt=32_32_32(13). */
    srd.w = (4u) | (5u << 3) | (6u << 6) | (7u << 9) | (4u << 12) | (13u << 15);
    uint32_t voffset = (uint32_t)(i * 12);
    __asm__ volatile(
        "buffer_store_format_xyz %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : : "v"(data), "v"(voffset), "s"(srd) : "memory");
}

int main(void) {
    uint32_t *h_src=(uint32_t*)malloc(N*3*4), *h_buf=(uint32_t*)malloc(N*3*4);
    for (int i=0;i<N*3;i++) h_src[i]=(uint32_t)(0x9e3779b9u*(uint32_t)(i+1));
    uint32_t *d_src,*d_buf;
    GPU_CHECK(hipMalloc(&d_src,N*3*4)); GPU_CHECK(hipMalloc(&d_buf,N*3*4));
    GPU_CHECK(hipMemset(d_buf,0,N*3*4));
    GPU_CHECK(hipMemcpy(d_src,h_src,N*3*4,hipMemcpyHostToDevice));
    int g=N/BLOCK_SIZE;
    hipLaunchKernelGGL(kernel,dim3(g),dim3(BLOCK_SIZE),0,0,d_src,d_buf,N); GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_buf,d_buf,N*3*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N*3;i++){ if(h_buf[i]!=h_src[i]){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_buf[i],h_src[i]);pass=0;break;} }
    if(pass) printf("buf[0]=0x%08x buf[3]=0x%08x\n",h_buf[0],h_buf[3]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
