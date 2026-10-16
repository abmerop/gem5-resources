/*
 * HIP test for SCRATCH_LOAD_DWORDX3 (ENC_FLAT/scratch, load 3 dwords).
 * Each lane round-trips 3 dwords through its private scratch region:
 * scratch_store_dwordx3 writes them, scratch_load_dwordx3 reads them back.
 * Verified against the stored values.  Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*data,const int*idx,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    volatile int pad[16]; int sel=idx[i]&15; pad[sel]=i;
    u32x3 vin; vin.x=data[i*3+0]; vin.y=data[i*3+1]; vin.z=data[i*3+2];
    int voff=0; u32x3 vout;
    __asm__ volatile(
        "scratch_store_dwordx3 %1, %2, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "scratch_load_dwordx3 %0, %1, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :"=v"(vout):"v"(voff),"v"(vin):"memory");
    out[i*3+0]=vout.x; out[i*3+1]=vout.y; out[i*3+2]=vout.z;
}
int main(void){
    uint32_t*h=(uint32_t*)malloc(N*3*4),*o=(uint32_t*)malloc(N*3*4);
    int*idx=(int*)malloc(N*4);
    for(int i=0;i<N*3;i++) h[i]=(uint32_t)(i*2246822519u+374761393u);
    for(int i=0;i<N;i++) idx[i]=i;
    uint32_t*d,*dO; int*di;
    GPU_CHECK(hipMalloc(&d,N*3*4));GPU_CHECK(hipMalloc(&dO,N*3*4));GPU_CHECK(hipMalloc(&di,N*4));
    GPU_CHECK(hipMemcpy(d,h,N*3*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(di,idx,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d,di,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(o,dO,N*3*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N*3;i++) if(o[i]!=h[i]){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,o[i],h[i]);pass=0;break;}
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",o[0],N*3-1,o[N*3-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
