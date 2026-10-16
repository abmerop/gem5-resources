/*
 * HIP test for S_ATC_PROBE_BUFFER (ENC_SMEM, ATC/address-translation probe of
 * a buffer descriptor).  Pattern B.
 *
 * S_ATC_PROBE_BUFFER has no architecturally-visible data result; it hints the
 * ATC to pre-translate an address range described by a buffer descriptor
 * (SBASE = 4 SGPRs) plus an offset. The test emits the instruction against a
 * descriptor built from a real allocation and then verifies a subsequent
 * computation still completes (i.e. the instruction executed without aborting
 * the simulation).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o s_atc_probe_buffer \
 *        s_atc_probe_buffer.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*a,const uint32_t*b,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    /* Build a 128-bit "buffer descriptor" from the out pointer; only used as
     * the SBASE quad for the probe.  probe flags imm = 0, offset = 0. */
    unsigned __int128 desc=(unsigned __int128)(uint64_t)out;
    __asm__ volatile(
        "s_atc_probe_buffer 0, %0, 0x0\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        ::"s"(desc):"memory");
    out[i]=a[i]+b[i];
}
int main(void){
    uint32_t*h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_a[i]=(uint32_t)(i*2654435761u+13);h_b[i]=(uint32_t)(0x9e3779b9u*(i+1));}
    uint32_t*d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel,dim3(N/BLOCK_SIZE),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t e=h_a[i]+h_b[i];if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],e);pass=0;break;}}
    if(pass)printf("out[1]=0x%08x out[2]=0x%08x\n",h_out[1],h_out[2]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
