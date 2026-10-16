/*
 * HIP test for DS_ADD_F64 (ENC_DS, LDS double atomic add, no return).
 *
 * gem5 (ds.cc) routes DS_ADD_F64 through atomicExecute<VecElemF64>:
 *     MEM[ADDR] = MEM[ADDR] + DATA[0:1]     (f64, no value returned)
 * Each lane owns a distinct 8-byte LDS slot (no contention), so the final slot
 * value is init + data.  All values are small exact-representable doubles so
 * the addition is bit-exact.  Emitted via inline asm on the LDS byte offset.
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
kern(const double*init,const double*data,double*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ double lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    uint32_t addr=(uint32_t)(uintptr_t)(&lds[t]);
    double v=data[i];
    __asm__ volatile("ds_add_f64 %0, %1\n\ts_waitcnt lgkmcnt(0)\n\t"
                     ::"v"(addr),"v"(v):"memory");
    __syncthreads(); mem_out[i]=lds[t];
}
int main(void){
    double *hi=(double*)malloc(N*8),*hd=(double*)malloc(N*8),*ho=(double*)malloc(N*8);
    for(int i=0;i<N;i++){ hi[i]=(double)((i%23)-11)*0.25; hd[i]=(double)((i%13)-6)*0.5; }
    double *di,*dd,*d_o;
    GPU_CHECK(hipMalloc(&di,N*8));GPU_CHECK(hipMalloc(&dd,N*8));GPU_CHECK(hipMalloc(&d_o,N*8));
    GPU_CHECK(hipMemcpy(di,hi,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dd,hd,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,d_o,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){ double e=hi[i]+hd[i];
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %f exp %f\n",i,ho[i],e);pass=0;} }
    if(pass)printf("out[0]=%f out[255]=%f\n",ho[0],ho[255]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
