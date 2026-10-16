/*
 * HIP test for V_XNOR_B32 (VOP2 32-bit XNOR), with DPP and SDWA variants.
 *
 *   D = ~(S0 ^ S1)
 *
 * gem5 dispatches DPP/SDWA forms through vop2Helper.  Three kernels:
 *   base : v_xnor_b32_e64 vd, va, vb                 -> ~(a ^ b)
 *   dpp  : v_xnor_b32_dpp ... quad_perm:[1,0,3,2]    -> ~(a[swz] ^ b)
 *   sdwa : v_xnor_b32_sdwa ... DWORD identity selects -> ~(a ^ b)
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_xnor_b32 v_xnor_b32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kb(const uint32_t*a,const uint32_t*b,uint32_t*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint32_t va=a[i],vb=b[i],vd;__asm__ volatile("v_xnor_b32_e32 %0, %1, %2\n\t":"=v"(vd):"v"(va),"v"(vb));o[i]=vd;}
__global__ void kd(const uint32_t*a,const uint32_t*b,uint32_t*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint32_t va=a[i],vb=b[i],vd;__asm__ volatile("v_xnor_b32_dpp %0, %1, %2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(vd):"v"(va),"v"(vb));o[i]=vd;}
__global__ void ks(const uint32_t*a,const uint32_t*b,uint32_t*o,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;uint32_t va=a[i],vb=b[i],vd;__asm__ volatile("v_xnor_b32_sdwa %0, %1, %2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD\n\t":"=v"(vd):"v"(va),"v"(vb));o[i]=vd;}
int main(void){
    uint32_t*h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t*hb=(uint32_t*)malloc(N*4),*hd=(uint32_t*)malloc(N*4),*hs=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_a[i]=(uint32_t)(i*2654435761u+1u);h_b[i]=(uint32_t)(i*40503u+7u);}
    uint32_t*da,*db,*ob,*od,*os;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&ob,N*4));GPU_CHECK(hipMalloc(&od,N*4));GPU_CHECK(hipMalloc(&os,N*4));
    GPU_CHECK(hipMemcpy(da,h_a,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,h_b,N*4,hipMemcpyHostToDevice));
    int g=N/BLOCK_SIZE;
    kb<<<g,BLOCK_SIZE>>>(da,db,ob,N);GPU_CHECK(hipGetLastError());
    kd<<<g,BLOCK_SIZE>>>(da,db,od,N);GPU_CHECK(hipGetLastError());
    ks<<<g,BLOCK_SIZE>>>(da,db,os,N);GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(hb,ob,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hd,od,N*4,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(hs,os,N*4,hipMemcpyDeviceToHost));
    static const int perm[4]={1,0,3,2};int pass=1;
    for(int i=0;i<N&&pass;i++){
        uint32_t eb=~(h_a[i]^h_b[i]);int bl=(i&~3)+perm[i&3];uint32_t ed=~(h_a[bl]^h_b[i]);uint32_t es=~(h_a[i]^h_b[i]);
        if(hb[i]!=eb){fprintf(stderr,"BASE[%d] 0x%x exp 0x%x\n",i,hb[i],eb);pass=0;}
        else if(hd[i]!=ed){fprintf(stderr,"DPP[%d] 0x%x exp 0x%x\n",i,hd[i],ed);pass=0;}
        else if(hs[i]!=es){fprintf(stderr,"SDWA[%d] 0x%x exp 0x%x\n",i,hs[i],es);pass=0;}
    }
    if(pass)printf("base0=0x%x dpp0=0x%x sdwa0=0x%x\n",hb[0],hd[0],hs[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
