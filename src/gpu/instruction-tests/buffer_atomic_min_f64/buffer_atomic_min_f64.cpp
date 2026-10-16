/*
 * HIP test for BUFFER_ATOMIC_MIN_F64 (ENC_MUBUF, 64-bit float atomic min,
 * return-old).  MUBUF addresses through a 128-bit V# in 4 SGPRs; per-thread
 * byte offset (offen).  vdata ("+v") carries the operand in and old out.
 * Inputs are finite distinct doubles so min is exact.  mem=min(init,data),
 * old=init.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(double*mem,const double*data,double*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)mem;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    srd.w=0x00020000u;
    uint32_t voffset=(uint32_t)(i*8);
    double v=data[i];
    __asm__ volatile("buffer_atomic_min_f64 %0, %1, %2, 0 offen sc0\n\ts_waitcnt vmcnt(0)\n\t"
                     : "+v"(v) : "v"(voffset), "s"(srd) : "memory");
    old_out[i]=v;
}
int main(void){
    double*hi=(double*)malloc(N*8),*hd=(double*)malloc(N*8),*ho=(double*)malloc(N*8),*hm=(double*)malloc(N*8);
    for(int i=0;i<N;i++){ hi[i]=(double)(i-128)*1.5; hd[i]=(double)(64-i)*2.25; }
    double*dm,*dd,*doo;
    GPU_CHECK(hipMalloc(&dm,N*8));GPU_CHECK(hipMalloc(&dd,N*8));GPU_CHECK(hipMalloc(&doo,N*8));
    GPU_CHECK(hipMemcpy(dm,hi,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dd,hd,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dm,dd,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(hm,dm,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(ho,doo,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ double eo=hi[i], em=hi[i]<hd[i]?hi[i]:hd[i];
        if(ho[i]!=eo||hm[i]!=em){fprintf(stderr,"FAIL[%d] old=%g(e %g) mem=%g(e %g)\n",i,ho[i],eo,hm[i],em);pass=0;break;}}
    if(pass)printf("old[1]=%g mem[1]=%g\n",ho[1],hm[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
