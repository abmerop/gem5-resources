/*
 * HIP test for V_ADD_CO_U32 (VOP2 with carry-out, u32).
 * gem5 (vop2.cc): vdst = src0 + src1; vcc[lane] = (u64(src0)+u64(src1) >=
 * 2^32). The carry-out goes to an SGPR mask (sdst). Base (e64) form.
 * Exact unsigned integer -> bit-exact compare of sum and per-lane carry.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kern(const uint32_t*a,const uint32_t*b,uint32_t*out,uint32_t*cout,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s0=a[i],s1=b[i],d; uint64_t carry;
    __asm__ volatile("v_add_co_u32 %0, %1, %2, %3\n\t"
                     :"=v"(d),"=s"(carry):"v"(s0),"v"(s1));
    out[i]=d;
    cout[i]=(uint32_t)((carry>>(threadIdx.x))&1ull);
}

int main(void){
    uint32_t*ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4);
    uint32_t*ho=(uint32_t*)malloc(N*4),*hco=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){
        ha[i]=(uint32_t)(i*2654435761u);
        hb[i]=(uint32_t)(0xFFFFFFF0u - i*1013904223u);
    }
    uint32_t*da,*db,*doo,*dco;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));
    GPU_CHECK(hipMalloc(&doo,N*4));GPU_CHECK(hipMalloc(&dco,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,doo,dco,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hco,dco,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t es=ha[i]+hb[i];
        uint32_t ec=(((uint64_t)ha[i]+(uint64_t)hb[i])>=0x100000000ull)?1:0;
        if(ho[i]!=es||hco[i]!=ec){fprintf(stderr,"FAIL[%d] sum got %u exp %u carry got %u exp %u\n",i,ho[i],es,hco[i],ec);pass=0;break;} }
    if(pass)printf("out[0]=%u carry[0]=%u\n",ho[0],hco[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
