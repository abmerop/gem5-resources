/*
 * HIP test for V_CLREXCP (VOP1, clear wave exception state).
 *
 * gem5 (src/arch/amdgpu/common/insts/vop1.cc) models this as a no-op (it only
 * warns "exceptions are not yet tracked"), which matches the architectural
 * effect on visible register/memory state: V_CLREXCP alters only the exception
 * status, nothing observable to a normal program.
 *
 * Verification strategy: the kernel performs a known arithmetic computation,
 * executes V_CLREXCP in the middle (via inline asm, no operands), and stores
 * the result.  A pass proves the instruction decoded and executed without
 * disturbing surrounding computation or crashing the simulator.  Pattern A,
 * base only (no DPP/SDWA).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N=256, BLOCK_SIZE=64;

__global__ void
kernel(const int32_t *a, const int32_t *b, int32_t *out, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    int32_t x = a[i] + b[i];
    __asm__ volatile("v_clrexcp\n\t" ::: "memory");
    int32_t y = a[i] * 3 - b[i];
    out[i] = x ^ y;
}

int main(void)
{
    int32_t *ha=(int32_t*)malloc(N*4),*hb=(int32_t*)malloc(N*4),*ho=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ha[i]=i*7-13;hb[i]=i*i-5;}
    int32_t *da,*db,*dout;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dout,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dout,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dout,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){int32_t x=ha[i]+hb[i],y=ha[i]*3-hb[i],e=x^y;
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,ho[i],e);pass=0;break;}}
    if(pass)printf("out[1]=%d\n",ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
