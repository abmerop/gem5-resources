/*
 * HIP test for V_PRNG_B32 (VOP1 pseudo-random bit advance), pattern A.
 * Exercises base, DPP and SDWA encodings (vop1Helper -> variants supported).
 *
 * gem5 (src/arch/amdgpu/common/insts/vop1.cc):
 *   D.u = (S0.u << 1) ^ ((S0.u >> 31) & 1 ? 0xc5 : 0x00)   [32-bit LFSR step]
 *
 * Variants exercised (probed with llvm-mc -mcpu=gfx950):
 *   base : v_prng_b32       vdst, vsrc
 *   dpp  : v_prng_b32_dpp   vdst, vsrc quad_perm:[1,0,3,2] ... -> src of lane^1
 *   sdwa : v_prng_b32_sdwa  vdst, vsrc dst_sel:DWORD src0_sel:DWORD -> like base
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_prng_b32 v_prng_b32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kern_base(const uint32_t*a,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s=a[i],d;
    __asm__ volatile("v_prng_b32 %0, %1\n\t":"=v"(d):"v"(s));
    out[i]=d;
}
__global__ void kern_dpp(const uint32_t*a,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s=a[i],d;
    __asm__ volatile(
        "v_prng_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t"
        :"=v"(d):"v"(s));
    out[i]=d;
}
__global__ void kern_sdwa(const uint32_t*a,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s=a[i],d;
    __asm__ volatile(
        "v_prng_b32_sdwa %0, %1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src0_sel:DWORD\n\t"
        :"=v"(d):"v"(s));
    out[i]=d;
}

static uint32_t prng(uint32_t in){
    return (in<<1) ^ (((in>>31)&1u) ? 0xc5u : 0x00u);
}

int main(void){
    uint32_t*h_a=(uint32_t*)malloc(N*4);
    uint32_t*h_b=(uint32_t*)malloc(N*4),*h_d=(uint32_t*)malloc(N*4),*h_s=(uint32_t*)malloc(N*4);
    /* Mix of high-bit-set and clear values to exercise both xor branches. */
    for(int i=0;i<N;i++) h_a[i]=(uint32_t)(i*2654435761u+0x9e3779b9u);
    uint32_t*d_a,*d_b,*d_d,*d_s;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_d,N*4));GPU_CHECK(hipMalloc(&d_s,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    int grid=N/BLOCK_SIZE;
    kern_base<<<grid,BLOCK_SIZE>>>(d_a,d_b,N);
    kern_dpp <<<grid,BLOCK_SIZE>>>(d_a,d_d,N);
    kern_sdwa<<<grid,BLOCK_SIZE>>>(d_a,d_s,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_b,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_d,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_s,d_s,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N && pass;i++){
        uint32_t eb=prng(h_a[i]);
        uint32_t ed=prng(h_a[i^1]);   /* quad_perm pair swap */
        uint32_t es=prng(h_a[i]);
        if(h_b[i]!=eb){fprintf(stderr,"FAIL base[%d] got 0x%08x exp 0x%08x\n",i,h_b[i],eb);pass=0;break;}
        if(h_d[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got 0x%08x exp 0x%08x\n",i,h_d[i],ed);pass=0;break;}
        if(h_s[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got 0x%08x exp 0x%08x\n",i,h_s[i],es);pass=0;break;}
    }
    if(pass)printf("base[1]=0x%08x dpp[1]=0x%08x sdwa[1]=0x%08x\n",h_b[1],h_d[1],h_s[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
