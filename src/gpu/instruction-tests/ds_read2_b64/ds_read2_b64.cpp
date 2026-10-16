/*
 * HIP test for DS_READ2_B64 (ENC_DS, dual 64-bit LDS read).
 *
 *   VDST[0:1] = LDS[addr + offset0*8]
 *   VDST[2:3] = LDS[addr + offset1*8]
 *
 * Each thread owns a 2x b64 (4-dword) LDS slot.  We fill both, read them back
 * with offset0:0 offset1:1, and verify.  Pattern C, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_read2_b64 ds_read2_b64.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

typedef uint64_t u64x2 __attribute__((ext_vector_type(2)));

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kernel_ds_read2_b64(const uint64_t *data, uint64_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    __shared__ uint64_t lds[BLOCK_SIZE * 2];
    int t = threadIdx.x;
    lds[t*2+0] = data[i*2+0];
    lds[t*2+1] = data[i*2+1];
    __syncthreads();
    __attribute__((address_space(3))) uint64_t *lp =
        (__attribute__((address_space(3))) uint64_t *)(&lds[t*2]);
    uint32_t addr = (uint32_t)(uintptr_t)lp;
    u64x2 vd;
    __asm__ volatile(
        "ds_read2_b64 %0, %1 offset0:0 offset1:1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "=v"(vd) : "v"(addr) : "memory");
    out[i*2+0] = vd.x;
    out[i*2+1] = vd.y;
}

int main(void)
{
    uint64_t *h_data=(uint64_t*)malloc(N*2*sizeof(uint64_t));
    uint64_t *h_out =(uint64_t*)malloc(N*2*sizeof(uint64_t));
    for(int i=0;i<N*2;i++) h_data[i]=(uint64_t)(i*0x9e3779b97f4a7c15ULL+17ULL);
    uint64_t *d_data,*d_out;
    GPU_CHECK(hipMalloc(&d_data,N*2*sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_out, N*2*sizeof(uint64_t)));
    GPU_CHECK(hipMemcpy(d_data,h_data,N*2*sizeof(uint64_t),hipMemcpyHostToDevice));
    kernel_ds_read2_b64<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_data,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*2*sizeof(uint64_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N*2;i++){
        if(h_out[i]!=h_data[i]){
            fprintf(stderr,"FAIL[%d] got 0x%llx exp 0x%llx\n",i,
                (unsigned long long)h_out[i],(unsigned long long)h_data[i]);pass=0;break;
        }
    }
    if(pass)printf("out[0]=0x%llx out[%d]=0x%llx\n",
        (unsigned long long)h_out[0],N*2-1,(unsigned long long)h_out[N*2-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
