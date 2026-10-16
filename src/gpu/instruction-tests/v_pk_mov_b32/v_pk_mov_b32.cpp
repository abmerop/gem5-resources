/*
 * HIP test for V_PK_MOV_B32 (VOP3P, 64-bit packed move of two dwords).
 * gem5 (vop3p.cc), OPSEL=0: dst.lo = src0[31:0], dst.hi = src1[31:0], i.e.
 * vdst64 = (src1.lo << 32) | src0.lo.  Base only (no DPP/SDWA form).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const uint64_t*a,const uint64_t*b,uint64_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t s0=a[i], s1=b[i], d;
    __asm__ volatile("v_pk_mov_b32 %0, %1, %2\n\t":"=v"(d):"v"(s0),"v"(s1));
    out[i]=d;
}
int main(void){
    uint64_t*ha=(uint64_t*)malloc(N*8),*hb=(uint64_t*)malloc(N*8),*ho=(uint64_t*)malloc(N*8);
    for(int i=0;i<N;i++){ ha[i]=0x1111000000000000ull|(uint64_t)(i*2654435761u);
                          hb[i]=0x2222000000000000ull|(uint64_t)((i*40503u)+7u); }
    uint64_t*da,*db,*doo;
    GPU_CHECK(hipMalloc(&da,N*8));GPU_CHECK(hipMalloc(&db,N*8));GPU_CHECK(hipMalloc(&doo,N*8));
    GPU_CHECK(hipMemcpy(da,ha,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(db,hb,N*8,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint64_t e=((hb[i]&0xffffffffull)<<32)|(ha[i]&0xffffffffull);
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got 0x%llx exp 0x%llx\n",i,(unsigned long long)ho[i],(unsigned long long)e);pass=0;break;}}
    if(pass)printf("out[1]=0x%llx\n",(unsigned long long)ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
