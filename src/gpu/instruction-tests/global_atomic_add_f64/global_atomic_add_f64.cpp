/*
 * HIP test for GLOBAL_ATOMIC_ADD_F64 (ENC_FLAT_GLBL, 64-bit float atomic add).
 * Decodes to Inst_FLAT__FLAT_ATOMIC_ADD_F64 in gem5 (atomicExecute<F64> add).
 * unsafeAtomicAdd on a per-thread global double slot (no contention) forces the
 * hardware global_atomic_add_f64; return discarded -> non-return form.
 * Values are small integers exactly representable in f64, so init+data is exact.
 * Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(double*mem,const double*data,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    unsafeAtomicAdd(&mem[i], data[i]);
}
int main(void){
    double*h_init=(double*)malloc(N*8),*h_data=(double*)malloc(N*8),*h_out=(double*)malloc(N*8);
    for(int i=0;i<N;i++){ h_init[i]=(double)(i-128); h_data[i]=(double)(2*i+3); }
    double*d_mem,*d_data;
    GPU_CHECK(hipMalloc(&d_mem,N*8)); GPU_CHECK(hipMalloc(&d_data,N*8));
    GPU_CHECK(hipMemcpy(d_mem,h_init,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_data,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_mem,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ double e=h_init[i]+h_data[i];
        if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,h_out[i],e);pass=0;break;}}
    if(pass)printf("out[0]=%g out[%d]=%g\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
