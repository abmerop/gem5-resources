/*
 * HIP test for DS_WRITE_ADDTID_B32 (ENC_DS, LDS write at m0_base + lane*4).
 * gem5 (ds.cc): addr[lane] = (m0 & 0xffff) + lane*4; LDS[addr] = DATA[lane].
 * Each lane writes its data via ds_write_addtid_b32 (m0 = byte offset of
 * lds[0]) into lds[lane], then reads lds[lane] back normally.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const uint32_t*in,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=0xFFFFFFFFu; __syncthreads();
    uint32_t base=0; // (uint32_t)(uintptr_t)(__attribute__((address_space(3)))uint32_t*)(&lds[0]);
    uint32_t d=in[i];
    __asm__ volatile("s_mov_b32 m0, %1\n\tds_write_addtid_b32 %0\n\ts_waitcnt lgkmcnt(0)\n\t"
                     : : "v"(d), "s"(base) : "m0","memory");
    __syncthreads();
    out[i]=lds[t];
}
int main(void){
    uint32_t*h_in=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_in[i]=(uint32_t)(i*40503u+0xbeadu);
    uint32_t*d_in,*d_out;
    GPU_CHECK(hipMalloc(&d_in,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++) if(h_out[i]!=h_in[i]){
        fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],h_in[i]);pass=0;break;}
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
