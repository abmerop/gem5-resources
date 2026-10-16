/*
 * HIP test for DS_READ_U16 (ENC_DS, LDS load unsigned 16-bit, zero-extend to
 * 32).  VDST.u32 = {16'b0, LDS[addr][15:0]}.  Each lane owns a private LDS
 * slot: a normal store places a value, then ds_read_u16 reads it back
 * zero-extended.  Verified against (uint32_t)(uint16_t)src[i].
 * Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint16_t*src,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint16_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=src[i]; __syncthreads();
    __attribute__((address_space(3))) uint16_t*lp=(__attribute__((address_space(3))) uint16_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint32_t dst=0xDEADBEEFu;
    __asm__ volatile(
        "ds_read_u16 %0, %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"=v"(dst):"v"(addr):"memory");
    out[i]=dst;
}
int main(void){
    uint16_t*h_src=(uint16_t*)malloc(N*2); uint32_t*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_src[i]=(uint16_t)(i*1237+5);
    uint16_t*d_src; uint32_t*d_out;
    GPU_CHECK(hipMalloc(&d_src,N*2));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_src,h_src,N*2,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t exp=(uint32_t)h_src[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%u out[%d]=%u\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
