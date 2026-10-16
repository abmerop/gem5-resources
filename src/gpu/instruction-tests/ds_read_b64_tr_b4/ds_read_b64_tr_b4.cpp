/*
 * HIP test for DS_READ_B64_TR_B4 (ENC_DS, 64-bit LDS transposing read of 4-bit
 * elements).
 *
 * Per the gem5 model (src/arch/amdgpu/common/insts/ds.cc), each lane reads a
 * 64-bit qword (sixteen 4-bit nibbles) from its LDS address, and the hardware
 * transposes nibbles within each group of 16 lanes.  A 64-lane wavefront is
 * split into 4 passes (lane groups p*16 .. p*16+15).  Numbering nibbles from
 * the MSB (nibble 0 = bits[63:60], nibble 15 = bits[3:0]), the result is:
 *
 *   out_qword[p*16 + i], nibble m  =  in_qword[p*16 + m], nibble i
 *
 * i.e. a 16x16 nibble transpose within each pass group.  This test seeds each
 * lane's LDS qword with a distinct known value, performs the transposing read,
 * and verifies every output nibble against the transpose formula.  Pattern C.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void
kernel(const uint64_t *data, uint64_t *out, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    __shared__ uint64_t lds[BLOCK_SIZE];
    int t = threadIdx.x;
    lds[t] = data[i];
    __syncthreads();
    __attribute__((address_space(3))) uint64_t *lp =
        (__attribute__((address_space(3))) uint64_t *)(&lds[t]);
    uint32_t addr = (uint32_t)(uintptr_t)lp;
    uint64_t vd;
    __asm__ volatile(
        "ds_read_b64_tr_b4 %0, %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        : "=v"(vd) : "v"(addr) : "memory");
    out[i] = vd;
}

static inline uint32_t nib(uint64_t q, int n) { return (uint32_t)((q >> (60 - 4*n)) & 0xf); }

int main(void)
{
    uint64_t *h_data=(uint64_t*)malloc((size_t)N*8),*h_out=(uint64_t*)malloc((size_t)N*8);
    for(int i=0;i<N;i++) h_data[i]=(uint64_t)i*0x0123456789abcdefULL + 0x1111111111111111ULL*(uint64_t)(i+1);
    uint64_t *d_data,*d_out;
    GPU_CHECK(hipMalloc(&d_data,(size_t)N*8));GPU_CHECK(hipMalloc(&d_out,(size_t)N*8));
    GPU_CHECK(hipMemcpy(d_data,h_data,(size_t)N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_data,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,(size_t)N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N && pass;i++){
        int b=i/BLOCK_SIZE, lane=i%BLOCK_SIZE, p=lane/16, ii=lane%16;
        uint64_t exp=0;
        for(int m=0;m<16;m++){
            uint64_t src=h_data[b*BLOCK_SIZE + p*16 + m];
            uint32_t v=nib(src, ii);
            exp |= (uint64_t)v << (60 - 4*m);
        }
        if(h_out[i]!=exp){
            fprintf(stderr,"FAIL[%d] got 0x%016llx exp 0x%016llx\n",i,
                (unsigned long long)h_out[i],(unsigned long long)exp);pass=0;}
    }
    if(pass)printf("out[0]=0x%016llx out[17]=0x%016llx\n",
        (unsigned long long)h_out[0],(unsigned long long)h_out[17]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
