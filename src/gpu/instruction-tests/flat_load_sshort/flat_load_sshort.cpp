/*
 * HIP test for FLAT_LOAD_SSHORT (ENC_FLAT, load signed 16-bit, sext to 32).
 *   VDST.i32 = sext( MEM[addr][15:0] )
 * Each thread loads one signed short from a global array via a flat address.
 * Verified against (int32_t)(int16_t)data[i].  Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const int16_t*data,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(&data[i]);
    int32_t d;
    __asm__ volatile(
        "flat_load_sshort %0, %1\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        :"=v"(d):"v"(addr):"memory");
    out[i]=d;
}
int main(void){
    int16_t*h_data=(int16_t*)malloc(N*2); int32_t*h_out=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_data[i]=(int16_t)(i*613-20000);
    int16_t*d_data; int32_t*d_out;
    GPU_CHECK(hipMalloc(&d_data,N*2));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_data,h_data,N*2,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_data,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){int32_t exp=(int32_t)h_data[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
