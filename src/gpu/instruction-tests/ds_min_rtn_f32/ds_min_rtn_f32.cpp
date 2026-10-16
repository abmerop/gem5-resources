/*
 * HIP test for DS_MIN_RTN_F32 (ENC_DS, LDS float atomic min, returning).
 *
 * gem5 (ds.cc) routes DS_MIN_RTN_F32 through atomicExecute<VecElemF32> with
 * the AtomicMin flag:
 *     tmp = MEM[ADDR]; MEM[ADDR] = min(tmp, DATA); RETURN_DATA = tmp
 * Each lane owns a distinct 4-byte LDS slot (no contention): returned old ==
 * init, final slot == min(init, data).  Distinct finite f32 (no NaN).
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
kern(const float*init,const float*data,float*old_out,float*mem_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ float lds[BLOCK_SIZE]; int t=threadIdx.x;
    lds[t]=init[i]; __syncthreads();
    uint32_t addr=(uint32_t)(uintptr_t)(&lds[t]);
    float v=data[i], old;
    __asm__ volatile("ds_min_rtn_f32 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t"
                     :"=v"(old):"v"(addr),"v"(v):"memory");
    __syncthreads(); old_out[i]=old; mem_out[i]=lds[t];
}
int main(void){
    float *hi=(float*)malloc(N*4),*hd=(float*)malloc(N*4),
          *ho=(float*)malloc(N*4),*hm=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ hi[i]=(float)((i%17)-8)*0.25f; hd[i]=(float)((i%11)-5)*0.5f; }
    float *di,*dd,*d_o,*d_m;
    GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dd,N*4));
    GPU_CHECK(hipMalloc(&d_o,N*4));GPU_CHECK(hipMalloc(&d_m,N*4));
    GPU_CHECK(hipMemcpy(di,hi,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dd,hd,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dd,d_o,d_m,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,d_o,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hm,d_m,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){ float eo=hi[i], em=hi[i]<hd[i]?hi[i]:hd[i];
        if(ho[i]!=eo){fprintf(stderr,"FAIL old[%d] got %g exp %g\n",i,ho[i],eo);pass=0;}
        else if(hm[i]!=em){fprintf(stderr,"FAIL mem[%d] got %g exp %g\n",i,hm[i],em);pass=0;} }
    if(pass)printf("old[1]=%g mem[1]=%g\n",ho[1],hm[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
