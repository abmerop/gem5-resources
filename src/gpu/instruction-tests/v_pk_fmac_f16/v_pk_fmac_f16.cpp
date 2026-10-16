/*
 * HIP test for V_PK_FMAC_F16 (VOP2 packed f16 multiply-accumulate).
 * gem5 (vop2.cc): for each of the two f16 lanes,
 *   D = fplibMulAdd(D, S0, S1) = D + S0*S1 (fused); vdst is the accumulator.
 * Panics on DPP/SDWA -> base only. Small integer f16 halves -> exact.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kern(const uint32_t*a,const uint32_t*b,const uint32_t*c,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s0=a[i],s1=b[i],d=c[i];
    __asm__ volatile("v_pk_fmac_f16 %0, %1, %2\n\t":"+v"(d):"v"(s0),"v"(s1));
    out[i]=d;
}

static uint16_t hbits(float f){ __half h=__float2half(f); uint16_t u; memcpy(&u,&h,2); return u; }
static uint32_t pk(float lo,float hi){ return ((uint32_t)hbits(hi)<<16)|hbits(lo); }

int main(void){
    uint32_t*ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4),*hc=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
    // All operands positive and nonzero so no product/sum is an exact zero;
    // this avoids the f16 fused signed-zero quirk (fplibMulAdd yields -0.0 for
    // cases like 0 + (-1)*0). Small ints keep everything exact in f16.
    int A0[N],A1[N],B0[N],B1[N],C0[N],C1[N];
    for(int i=0;i<N;i++){
        A0[i]=(i%4)+1;A1[i]=(i%3)+1;B0[i]=(i%5)+1;B1[i]=(i%4)+1;C0[i]=(i%6)+1;C1[i]=(i%7)+1;
        ha[i]=pk((float)A0[i],(float)A1[i]);
        hb[i]=pk((float)B0[i],(float)B1[i]);
        hc[i]=pk((float)C0[i],(float)C1[i]);
    }
    uint32_t*da,*db,*dc,*doo;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));
    GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t e=pk((float)(C0[i]+A0[i]*B0[i]),(float)(C1[i]+A1[i]*B1[i]));
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,ho[i],e);pass=0;break;} }
    if(pass)printf("out[0]=0x%08x out[100]=0x%08x\n",ho[0],ho[100]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
