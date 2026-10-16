/*
 * HIP test for DS_PK_ADD_RTN_F16 (ENC_DS, LDS packed-f16 atomic add, return).
 *
 * gem5 routes this through the packed-f16 atomic (AtomicOpPkAddF16):
 *     tmp = MEM[ADDR];
 *     MEM.f16lo += DATA.f16lo;  MEM.f16hi += DATA.f16hi;
 *     RETURN_DATA = tmp
 * Each lane owns a distinct 4-byte LDS slot (no contention): returned old ==
 * init, final slot == packed(init+data).  Exact integer f16 values.
 * Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void
kern(const uint32_t*init,const uint32_t*data,uint32_t*old_out,uint32_t*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    uint32_t addr=(uint32_t)(uintptr_t)(&lds[t]);
    uint32_t v=data[i], old;
    __asm__ volatile("ds_pk_add_rtn_f16 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t"
                     :"=v"(old):"v"(addr),"v"(v):"memory");
    __syncthreads(); old_out[i]=old; mem_out[i]=lds[t];
}
static uint16_t hb(float v){ __half h=__float2half(v); uint16_t b; __builtin_memcpy(&b,&h,2); return b; }
static float hv(uint16_t b){ __half h; __builtin_memcpy(&h,&b,2); return __half2float(h); }
static uint32_t pack(uint16_t lo,uint16_t hi){ return (uint32_t)lo | ((uint32_t)hi<<16); }
int main(void){
    uint32_t *hi_=(uint32_t*)malloc(N*4),*hd=(uint32_t*)malloc(N*4),
             *ho=(uint32_t*)malloc(N*4),*hm=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){
        hi_[i]=pack(hb((float)(i%7)), hb((float)((i%5)+1)));
        hd[i] =pack(hb((float)((i%3)+2)), hb((float)(i%9)));
    }
    uint32_t *di,*dd,*d_o,*d_m;
    GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dd,N*4));
    GPU_CHECK(hipMalloc(&d_o,N*4));GPU_CHECK(hipMalloc(&d_m,N*4));
    GPU_CHECK(hipMemcpy(di,hi_,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dd,hd,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,d_o,d_m,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,d_o,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hm,d_m,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){
        uint16_t ilo=hi_[i]&0xffff, ihi=hi_[i]>>16, dlo=hd[i]&0xffff, dhi=hd[i]>>16;
        uint32_t em=pack(hb(hv(ilo)+hv(dlo)), hb(hv(ihi)+hv(dhi)));
        if(ho[i]!=hi_[i]){fprintf(stderr,"FAIL old[%d] got %08x exp %08x\n",i,ho[i],hi_[i]);pass=0;}
        else if(hm[i]!=em){fprintf(stderr,"FAIL mem[%d] got %08x exp %08x\n",i,hm[i],em);pass=0;}
    }
    if(pass)printf("old[1]=%08x mem[1]=%08x\n",ho[1],hm[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
