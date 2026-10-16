/*
 * HIP test for SCRATCH_LOAD_USHORT (ENC_FLAT/scratch, load unsigned 16-bit,
 * zero-extend to 32).  Each lane round-trips a value through its private
 * scratch space: scratch_store_short writes 16 bits, scratch_load_ushort reads
 * them back zero-extended.  Verified against (uint32_t)(uint16_t)in[i].
 * Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint16_t*in,const int*idx,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    volatile int pad[8]; int sel=idx[i]&7; pad[sel]=i;
    int val=(int)in[i]; int voff=16; uint32_t result;
    __asm__ volatile(
        "scratch_store_short %1, %2, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        "scratch_load_ushort %0, %1, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :"=v"(result):"v"(voff),"v"(val):"memory");
    out[i]=result;
}
int main(void){
    uint16_t*h_in=(uint16_t*)malloc(N*2); int*h_idx=(int*)malloc(N*4); uint32_t*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_in[i]=(uint16_t)(i*997+3);h_idx[i]=i;}
    uint16_t*d_in; int*d_idx; uint32_t*d_out;
    GPU_CHECK(hipMalloc(&d_in,N*2));GPU_CHECK(hipMalloc(&d_idx,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_in,h_in,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_idx,h_idx,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_idx,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint32_t exp=(uint32_t)h_in[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%u out[%d]=%u\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
