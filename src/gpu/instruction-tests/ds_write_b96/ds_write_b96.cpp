/*
 * HIP test for DS_WRITE_B96 (ENC_DS, 96-bit LDS store).
 * Stores 3 consecutive dwords from the data VGPR range to LDS:
 *   LDS[addr .. addr+11] = DATA[95:0].
 * Each lane owns a distinct 12-byte LDS slot, pre-filled with a sentinel.  The
 * value is stored via forced ds_write_b96, read back with normal loads, and
 * verified.  Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*data,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE*3]; int t=threadIdx.x;
    for(int k=0;k<3;k++) lds[t*3+k]=0xEEEEEEEEu; __syncthreads();
    __attribute__((address_space(3))) uint32_t*lp=(__attribute__((address_space(3))) uint32_t*)(&lds[t*3]);
    uint32_t addr=(uint32_t)(uintptr_t)lp;
    u32x3 vdata; vdata.x=data[i*3+0]; vdata.y=data[i*3+1]; vdata.z=data[i*3+2];
    __asm__ volatile(
        "ds_write_b96 %0, %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        ::"v"(addr),"v"(vdata):"memory");
    __syncthreads();
    for(int k=0;k<3;k++) out[i*3+k]=lds[t*3+k];
}
int main(void){
    uint32_t*h=(uint32_t*)malloc(N*3*4),*o=(uint32_t*)malloc(N*3*4);
    for(int i=0;i<N*3;i++) h[i]=(uint32_t)(i*2654435761u+1013904223u);
    uint32_t*d,*dO;
    GPU_CHECK(hipMalloc(&d,N*3*4));GPU_CHECK(hipMalloc(&dO,N*3*4));
    GPU_CHECK(hipMemcpy(d,h,N*3*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(o,dO,N*3*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N*3;i++) if(o[i]!=h[i]){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,o[i],h[i]);pass=0;break;}
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",o[0],N*3-1,o[N*3-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
