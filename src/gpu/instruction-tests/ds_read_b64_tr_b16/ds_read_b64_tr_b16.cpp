/*
 * HIP test for DS_READ_B64_TR_B16 (ENC_DS, 64-bit LDS transposing read of
 * 16-bit elements).  Mirrors the gem5 completeAcc model (ds.cc): a 64-lane
 * wavefront is processed in 4 passes of 16 lanes.  Numbering 16-bit elements
 * from the MSB (e=0 -> bits[63:48] ... e=3 -> bits[15:0]), the result for output
 * lane (p*16+i) has, at element x (x=0..3):
 *     out.elem(x) = in_qword(p*16 + x*4 + i/4).elem(i%4)
 * Each lane seeds its own LDS qword; verified per element.  Variants: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint64_t*data,uint64_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint64_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=data[i]; __syncthreads();
    __attribute__((address_space(3))) uint64_t*lp=
        (__attribute__((address_space(3))) uint64_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp; uint64_t vd;
    __asm__ volatile("ds_read_b64_tr_b16 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"
                     :"=v"(vd):"v"(addr):"memory");
    out[i]=vd;
}
static inline uint16_t el16(uint64_t q,int e){ return (uint16_t)((q>>(48-16*e))&0xffffULL); }
int main(void){
    uint64_t*hd=(uint64_t*)malloc((size_t)N*8),*ho=(uint64_t*)malloc((size_t)N*8);
    for(int i=0;i<N;i++) hd[i]=(uint64_t)i*0x0123456789abcdefULL+0x1111111111111111ULL*(uint64_t)(i+1);
    uint64_t*dd,*dO; GPU_CHECK(hipMalloc(&dd,(size_t)N*8));GPU_CHECK(hipMalloc(&dO,(size_t)N*8));
    GPU_CHECK(hipMemcpy(dd,hd,(size_t)N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dd,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,(size_t)N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N && pass;i++){
        int b=i/BLOCK_SIZE,lane=i%BLOCK_SIZE,p=lane/16,ii=lane%16;
        uint64_t exp=0;
        for(int x=0;x<4;x++){
            uint64_t src=hd[b*BLOCK_SIZE + p*16 + x*4 + ii/4];
            uint16_t v=el16(src, ii%4);
            exp |= (uint64_t)v << (48-16*x);
        }
        if(ho[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%016llx exp 0x%016llx\n",i,
            (unsigned long long)ho[i],(unsigned long long)exp);pass=0;}
    }
    if(pass)printf("out[0]=0x%016llx out[17]=0x%016llx\n",
        (unsigned long long)ho[0],(unsigned long long)ho[17]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
