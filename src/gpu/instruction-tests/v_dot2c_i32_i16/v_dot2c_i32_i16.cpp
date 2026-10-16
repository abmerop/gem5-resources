/*
 * HIP test for V_DOT2C_I32_I16 (VOP2, i16x2 dot accumulate into i32).
 * gem5 (vop2.cc): vdst += sext16(S0.lo)*sext16(S1.lo)
 *                       + sext16(S0.hi)*sext16(S1.hi)   (vdst is accumulator)
 * SDWA panics; DPP is supported (swizzles SRC0 only). Signed 16-bit products
 * accumulated into i32 -> exact integer -> bit-exact compare.
 *
 * Variants exercised: base, dpp.
 *   For the DPP kernel SRC0 is held constant within each group of 4 lanes so
 *   quad_perm swizzling of SRC0 is a no-op; the reference then matches base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kern_base(const uint32_t*a,const uint32_t*b,const int32_t*c,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s0=a[i],s1=b[i]; int32_t d=c[i];
    __asm__ volatile("v_dot2c_i32_i16_e32 %0, %1, %2\n\t":"+v"(d):"v"(s0),"v"(s1));
    out[i]=d;
}
__global__ void kern_dpp(const uint32_t*a,const uint32_t*b,const int32_t*c,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s0=a[i],s1=b[i]; int32_t d=c[i];
    __asm__ volatile("v_dot2c_i32_i16_dpp %0, %1, %2 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t":"+v"(d):"v"(s0),"v"(s1));
    out[i]=d;
}

static inline int32_t s16(uint32_t v){ return (int32_t)(int16_t)(v & 0xffff); }
static int32_t ref(uint32_t s0,uint32_t s1,int32_t acc){
    acc += s16(s0)*s16(s1);
    acc += s16(s0>>16)*s16(s1>>16);
    return acc;
}
static uint32_t pk(int16_t lo,int16_t hi){ return ((uint32_t)(uint16_t)hi<<16)|(uint16_t)lo; }

int main(void){
    uint32_t*ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4),*had=(uint32_t*)malloc(N*4);
    int32_t*hc=(int32_t*)malloc(N*4),*ho=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++){
        ha[i]=pk((int16_t)((i%13)-6),(int16_t)((i%7)-3));           // per-lane src0 (base)
        had[i]=pk((int16_t)((i/4)%13-6),(int16_t)((i/4)%7-3));      // per-quad-const src0 (dpp)
        hb[i]=pk((int16_t)((i%9)-4),(int16_t)((i%11)-5));
        hc[i]=(int32_t)((i%23)-11);
    }
    uint32_t*da,*db,*dad; int32_t*dc,*doo;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dad,N*4));
    GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dad,had,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));

    int pass=1;
    // --- base ---
    kern_base<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    for(int i=0;i<N;i++){ int32_t e=ref(ha[i],hb[i],hc[i]);
        if(ho[i]!=e){fprintf(stderr,"BASE FAIL[%d] got %d exp %d\n",i,ho[i],e);pass=0;break;} }
    // --- dpp ---
    if(pass){
        kern_dpp<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dad,db,dc,doo,N);
        GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
        GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
        for(int i=0;i<N;i++){ int32_t e=ref(had[i],hb[i],hc[i]);
            if(ho[i]!=e){fprintf(stderr,"DPP FAIL[%d] got %d exp %d\n",i,ho[i],e);pass=0;break;} }
    }
    if(pass)printf("base+dpp ok out[100]=%d\n",ho[100]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
