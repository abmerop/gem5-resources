/*
 * HIP test for DS_CMPST_B64 (ENC_DS, LDS 64-bit compare-and-store, no return).
 *
 * gem5 (ds.cc cmpstExecute<U64>) semantics (DATA0=cmp, DATA1=src):
 *     tmp = MEM[ADDR];                 // 64b
 *     MEM[ADDR] = (tmp == cmp) ? src : tmp;      (no value returned)
 * Assembly: ds_cmpst_b64 addr, cmp[0:1], src[0:1].
 * Per-lane distinct 8-byte slot; half the lanes match (store src), half don't.
 * Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void
kern(const uint64_t*init,const uint64_t*cmp,const uint64_t*src,uint64_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint64_t lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    uint32_t addr=(uint32_t)(uintptr_t)(&lds[t]);
    uint64_t c=cmp[i], s=src[i];
    __asm__ volatile("ds_cmpst_b64 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t"
                     ::"v"(addr),"v"(c),"v"(s):"memory");
    __syncthreads(); out[i]=lds[t];
}
int main(void){
    uint64_t *hi=(uint64_t*)malloc(N*8),*hc=(uint64_t*)malloc(N*8),
             *hs=(uint64_t*)malloc(N*8),*ho=(uint64_t*)malloc(N*8);
    for(int i=0;i<N;i++){ hi[i]=0x100000000ULL+(uint64_t)i*7ULL;
        hs[i]=0xDEADBEEF00000000ULL+i; hc[i]=(i&1)?hi[i]:(hi[i]^0xFFULL); }
    uint64_t *di,*dc,*ds,*d_o;
    GPU_CHECK(hipMalloc(&di,N*8));GPU_CHECK(hipMalloc(&dc,N*8));
    GPU_CHECK(hipMalloc(&ds,N*8));GPU_CHECK(hipMalloc(&d_o,N*8));
    GPU_CHECK(hipMemcpy(di,hi,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dc,hc,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(ds,hs,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dc,ds,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,d_o,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){ uint64_t e=(hi[i]==hc[i])?hs[i]:hi[i];
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %016llx exp %016llx\n",i,
            (unsigned long long)ho[i],(unsigned long long)e);pass=0;} }
    if(pass)printf("out[0]=%016llx out[1]=%016llx\n",
        (unsigned long long)ho[0],(unsigned long long)ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
