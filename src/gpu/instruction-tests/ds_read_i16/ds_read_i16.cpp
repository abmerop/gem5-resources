/*
 * HIP test for DS_READ_I16 (ENC_DS, signed 16-bit LDS load, sign-extended).
 * gem5 (ds.cc): completeAcc reads a VecElemI16 from LDS and writes
 *   VDST.u32 = (int32_t)(int16_t)LDS[addr]     (sign-extended to 32 bits)
 * Each lane owns a private LDS slot into which a signed 16-bit value (incl
 * negatives) is stored; ds_read_i16 loads it back sign-extended.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const int16_t*in,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ int16_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=in[i]; __syncthreads();
    __attribute__((address_space(3))) int16_t*lp=
        (__attribute__((address_space(3))) int16_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; int32_t val;
    __asm__ volatile("ds_read_i16 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"
                     : "=v"(val) : "v"(addr) : "memory");
    out[i]=val;
}
int main(void){
    int16_t*h_in=(int16_t*)malloc(N*sizeof(int16_t));
    int32_t*h_out=(int32_t*)malloc(N*sizeof(int32_t));
    for(int i=0;i<N;i++) h_in[i]=(int16_t)(-30000 + i*250);
    int16_t*d_in; int32_t*d_out;
    GPU_CHECK(hipMalloc(&d_in,N*sizeof(int16_t)));GPU_CHECK(hipMalloc(&d_out,N*sizeof(int32_t)));
    GPU_CHECK(hipMemcpy(d_in,h_in,N*sizeof(int16_t),hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int32_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ int32_t exp=(int32_t)h_in[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
