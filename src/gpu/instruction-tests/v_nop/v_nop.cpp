/*
 * HIP test for V_NOP (VOP1 no-op).
 *
 * gem5 implements Inst_VOP1__V_NOP::execute as an empty body (a true no-op).
 * There is nothing to compute, so the test surrounds a v_nop with a real VALU
 * computation and verifies the computation is unaffected (i.e. v_nop neither
 * clobbers registers nor perturbs the pipeline).
 *
 * V_NOP has no operands, so the DPP/SDWA "variants" are not meaningful (there
 * is no source to swizzle / byte-select); only the base form is exercised.
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
kern(const uint32_t*a,const uint32_t*b,uint32_t*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t x=a[i], y=b[i], d;
    __asm__ volatile(
        "v_nop\n\t"
        "v_add_u32 %0, %1, %2\n\t"
        "v_nop\n\t"
        :"=v"(d):"v"(x),"v"(y));
    o[i]=d;
}
int main(void){
    uint32_t *ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ ha[i]=i*3+1; hb[i]=i*7+5; }
    uint32_t *da,*db,*do_;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&do_,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,do_,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,do_,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){ uint32_t e=ha[i]+hb[i];
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %u exp %u\n",i,ho[i],e);pass=0;} }
    if(pass)printf("out[1]=%u\n",ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
