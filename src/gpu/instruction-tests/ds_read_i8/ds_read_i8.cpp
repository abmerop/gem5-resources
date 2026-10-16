/*
 * HIP test for DS_READ_I8 (ENC_DS, signed-byte LDS read, sign-extended to 32b).
 *   VDST.i32 = sext8(LDS[addr].byte)
 * Each thread owns a 1-dword LDS slot; we seed its low byte and read it back.
 * Pattern C, base only.
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_read_i8 ds_read_i8.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*data,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=data[i]; __syncthreads();
    __attribute__((address_space(3))) uint32_t*lp=(__attribute__((address_space(3))) uint32_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; int32_t vd;
    __asm__ volatile("ds_read_i8 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t":"=v"(vd):"v"(addr):"memory");
    out[i]=vd;
}
int main(void){
    uint32_t*h_data=(uint32_t*)malloc(N*4); int32_t*h_out=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++)h_data[i]=(uint32_t)((i*7+0x33)&0xff) | 0xAB00u;  /* low byte varies, incl sign-bit set */
    uint32_t*dd;int32_t*dout;GPU_CHECK(hipMalloc(&dd,N*4));GPU_CHECK(hipMalloc(&dout,N*4));
    GPU_CHECK(hipMemcpy(dd,h_data,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dd,dout,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,dout,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){int32_t exp=(int32_t)(int8_t)(h_data[i]&0xff);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
