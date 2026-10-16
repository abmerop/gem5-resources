/*
 * HIP test for V_MAC_F16 (VOP2, f16 multiply-accumulate).
 * gem5 (vop2.cc): vdst = fplibMulAdd(D0, S0, S1) = D0 + S0*S1 (fused),
 * where D0 is the destination accumulator. DPP supported via vop2Helper
 * (SDWA not exercised here). Small integer f16 operands -> exact -> compare.
 *
 * Variants exercised: base, dpp.
 *   DPP kernel keeps SRC0 constant within each group of 4 lanes so quad_perm
 *   swizzling of SRC0 is a no-op and the reference matches base.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

__global__ void kern_base(const __half*a,const __half*b,const __half*c,__half*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __half s0=a[i],s1=b[i],d=c[i];
    __asm__ volatile("v_mac_f16_e32 %0, %1, %2\n\t":"+v"(d):"v"(s0),"v"(s1));
    out[i]=d;
}
__global__ void kern_dpp(const __half*a,const __half*b,const __half*c,__half*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __half s0=a[i],s1=b[i],d=c[i];
    __asm__ volatile("v_mac_f16_dpp %0, %1, %2 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t":"+v"(d):"v"(s0),"v"(s1));
    out[i]=d;
}

int main(void){
    __half *ha=(__half*)malloc(N*2),*had=(__half*)malloc(N*2),
           *hb=(__half*)malloc(N*2),*hc=(__half*)malloc(N*2),*ho=(__half*)malloc(N*2);
    int A[N],AD[N],B[N],C[N];
    for(int i=0;i<N;i++){
        A[i]=(i%9)-4; AD[i]=((i/4)%9)-4; B[i]=(i%7)-3; C[i]=(i%11)-5;
        ha[i]=__float2half((float)A[i]); had[i]=__float2half((float)AD[i]);
        hb[i]=__float2half((float)B[i]); hc[i]=__float2half((float)C[i]);
    }
    __half*da,*dad,*db,*dc,*doo;
    GPU_CHECK(hipMalloc(&da,N*2));GPU_CHECK(hipMalloc(&dad,N*2));GPU_CHECK(hipMalloc(&db,N*2));
    GPU_CHECK(hipMalloc(&dc,N*2));GPU_CHECK(hipMalloc(&doo,N*2));
    GPU_CHECK(hipMemcpy(da,ha,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dad,had,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(db,hb,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(dc,hc,N*2,hipMemcpyHostToDevice));

    int pass=1;
    kern_base<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*2,hipMemcpyDeviceToHost));
    for(int i=0;i<N;i++){ float e=(float)(C[i]+A[i]*B[i]); float g=__half2float(ho[i]);
        if(g!=e){fprintf(stderr,"BASE FAIL[%d] got %g exp %g\n",i,g,e);pass=0;break;} }
    if(pass){
        kern_dpp<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dad,db,dc,doo,N);
        GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
        GPU_CHECK(hipMemcpy(ho,doo,N*2,hipMemcpyDeviceToHost));
        for(int i=0;i<N;i++){ float e=(float)(C[i]+AD[i]*B[i]); float g=__half2float(ho[i]);
            if(g!=e){fprintf(stderr,"DPP FAIL[%d] got %g exp %g\n",i,g,e);pass=0;break;} }
    }
    if(pass)printf("base+dpp ok out[100]=%g\n",__half2float(ho[100]));
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
