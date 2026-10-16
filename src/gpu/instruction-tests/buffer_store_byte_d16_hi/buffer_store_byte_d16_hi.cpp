/*
 * HIP test for BUFFER_STORE_BYTE_D16_HI (MUBUF, gfx950 / CDNA4).
 *
 * Stores bits [23:16] of the source data VGPR as a byte to memory:
 *     stored_byte = (srcval >> 16) & 0xff
 * (mubuf.cc Inst_MUBUF__BUFFER_STORE_BYTE_D16_HI uses bits(data,23,16)).
 *
 * Each thread writes byte[23:16] of a known dword to its own per-thread byte
 * slot via a hand-built 128-bit buffer resource descriptor (V#).  Variants: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(uint8_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)out;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(4u<<12)|(4u<<15);
    uint32_t val=((uint32_t)((0x30+i)&0xff)<<16)|0x00000000u; val|=0x1200005Au;
    uint32_t voff=(uint32_t)i;
    __asm__ volatile("buffer_store_byte_d16_hi %0, %1, %2, 0 offen\n\ts_waitcnt vmcnt(0)\n\t"
                     ::"v"(val),"v"(voff),"s"(srd):"memory");
}
int main(void){
    uint8_t*ho=(uint8_t*)malloc(N);
    uint8_t*dO; GPU_CHECK(hipMalloc(&dO,N)); GPU_CHECK(hipMemset(dO,0x55,N));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){uint8_t exp=(uint8_t)((0x30+i)&0xff);
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%02x exp 0x%02x\n",i,ho[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%02x out[%d]=0x%02x\n",ho[0],N-1,ho[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
