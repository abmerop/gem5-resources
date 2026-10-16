/*
 * HIP test for V_SUBBREV_CO_U32 (VOP2 reverse subtract with borrow-in/out).
 * gem5 (vop2.cc): carry_in = vcc[lane];
 *   vdst  = src1 - src0 - carry_in;                 (u32 wrap)
 *   vcc_out[lane] = ((u32)(src0 + carry_in) > src1) ? 1 : 0;
 * Base (e64) form. The borrow-in is supplied via an SGPR mask (kernel arg).
 * Exact unsigned integer -> bit-exact compare of diff and per-lane borrow-out.
 * Operand order in asm: %2 = src0, %3 = src1; result = src1 - src0 - cin.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kern(const uint32_t*a,const uint32_t*b,uint64_t cin,
                     uint32_t*out,uint32_t*cout,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s0=a[i],s1=b[i],d; uint64_t co;
    __asm__ volatile("v_subbrev_co_u32 %0, %1, %2, %3, %4\n\t"
                     :"=v"(d),"=s"(co):"v"(s0),"v"(s1),"s"(cin));
    out[i]=d;
    cout[i]=(uint32_t)((co>>(threadIdx.x))&1ull);
}

int main(void){
    const uint64_t CIN=0xC3C3C3C3C3C3C3C3ull;
    uint32_t*ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4);
    uint32_t*ho=(uint32_t*)malloc(N*4),*hco=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ ha[i]=(uint32_t)(i*40503u+3); hb[i]=(uint32_t)(i*2654435761u); }
    uint32_t*da,*db,*doo,*dco;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));
    GPU_CHECK(hipMalloc(&doo,N*4));GPU_CHECK(hipMalloc(&dco,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,CIN,doo,dco,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hco,dco,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t cb=(uint32_t)((CIN>>(i%64))&1ull);
        uint32_t es=hb[i]-ha[i]-cb;                    // src1 - src0 - cin
        uint32_t ec=((uint32_t)(ha[i]+cb) > hb[i])?1:0; // (src0+cin) > src1
        if(ho[i]!=es||hco[i]!=ec){fprintf(stderr,"FAIL[%d] diff got %u exp %u borrow got %u exp %u\n",i,ho[i],es,hco[i],ec);pass=0;break;} }
    if(pass)printf("out[0]=%u borrow[0]=%u\n",ho[0],hco[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
