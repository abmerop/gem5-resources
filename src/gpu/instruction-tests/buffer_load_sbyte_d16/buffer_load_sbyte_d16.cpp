/*
 * HIP test for BUFFER_LOAD_SBYTE_D16 (ENC_MUBUF).
 * gem5 (mubuf.cc completeAcc): vdst[15:0] = {8'h0, MEM[addr].u8}; the other
 * half is not defined by this op, so we verify ONLY the loaded 16-bit half.
 * MUBUF raw V# (stride 0, numrec 0xffffffff, offen), 1-byte element stride.
 * Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const uint8_t*src,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)src;
    u32x4 srd; srd.x=(uint32_t)(base&0xffffffffu); srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu; srd.w=0x00020000u;
    uint32_t voff=(uint32_t)i; uint32_t d=0xC5C5C5C5u;
    __asm__ volatile("buffer_load_sbyte_d16 %0, %1, %2, 0 offen\n\ts_waitcnt vmcnt(0)\n\t"
                     : "+v"(d) : "v"(voff),"s"(srd) : "memory");
    out[i]=d;
}
int main(void){
    uint8_t*hs=(uint8_t*)malloc(N); uint32_t*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) hs[i]=(uint8_t)((i*37+5)&0xff);
    uint8_t*ds; uint32_t*doo;
    GPU_CHECK(hipMalloc(&ds,N));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(ds,hs,N,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(ds,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint16_t got=ho[i]&0xffff, e=(uint16_t)(int16_t)(int8_t)hs[i];
        if(got!=e){fprintf(stderr,"FAIL[%d] lo=%04x exp %04x (full %08x)\n",i,got,e,ho[i]);pass=0;break;}}
    if(pass)printf("out[1]=%08x lo=%04x\n",ho[1],ho[1]&0xffff);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
