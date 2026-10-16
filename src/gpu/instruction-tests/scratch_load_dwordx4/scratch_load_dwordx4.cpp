/*
 * HIP test for SCRATCH_LOAD_DWORDX4 (ENC_FLAT/scratch, load 4 dword(s)).
 * Each lane round-trips 4 dword(s) through private scratch:
 * scratch_store_dwordx4 writes, scratch_load_dwordx4 reads back.  Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*data,const int*idx,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    volatile int pad[16]; int sel=idx[i]&15; pad[sel]=i;
    int voff=0; u32x4 vin; vin.x=data[i*4+0]; vin.y=data[i*4+1]; vin.z=data[i*4+2]; vin.w=data[i*4+3]; u32x4 vout;
    __asm__ volatile(
        "scratch_store_dwordx4 %1, %2, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "scratch_load_dwordx4 %0, %1, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :"=v"(vout):"v"(voff),"v"(vin):"memory");
    out[i*4+0]=vout.x; out[i*4+1]=vout.y; out[i*4+2]=vout.z; out[i*4+3]=vout.w;
}
int main(void){
    const int W=4;
    uint32_t*h=(uint32_t*)malloc(N*W*4),*o=(uint32_t*)malloc(N*W*4);
    int*idx=(int*)malloc(N*4);
    for(int i=0;i<N*W;i++) h[i]=(uint32_t)(i*2246822519u+374761393u);
    for(int i=0;i<N;i++) idx[i]=i;
    uint32_t*d,*dO; int*di;
    GPU_CHECK(hipMalloc(&d,N*W*4));GPU_CHECK(hipMalloc(&dO,N*W*4));GPU_CHECK(hipMalloc(&di,N*4));
    GPU_CHECK(hipMemcpy(d,h,N*W*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(di,idx,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d,di,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(o,dO,N*W*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N*4;i++) if(o[i]!=h[i]){fprintf(stderr,"FAIL[%d] 0x%08x exp 0x%08x\n",i,o[i],h[i]);pass=0;break;}
    if(pass)printf("out[0]=0x%08x\n",o[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
