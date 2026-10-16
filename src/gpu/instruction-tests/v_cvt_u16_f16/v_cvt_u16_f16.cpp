/*
 * HIP test for V_CVT_U16_F16 (ENC_VOP1) plus DPP and SDWA variants.
 *
 * gem5 semantics (vop1.cc):
 *   D.u16 = (uint16_t) float(S0.f16)     (truncation toward zero)
 * The f16 source is converted to float and truncated to an unsigned 16-bit
 * integer.
 *
 * Inputs are non-negative f16 values with exact integer value (0..300) so the
 * truncated result is bit-exact.
 *
 * Variants exercised:
 *   base : v_cvt_u16_f16_e32  vd, va                 D = (u16)a[i]
 *   dpp  : v_cvt_u16_f16_dpp  ... quad_perm:[1,0,3,2] D = (u16)a[swizzled]
 *   sdwa : v_cvt_u16_f16_sdwa ... src0_sel:WORD_0     D = (u16)a[i]
 *
 * Variants: base,dpp,sdwa
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const uint16_t *a, uint16_t *o_base, uint16_t *o_dpp, uint16_t *o_sdwa, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t x = a[i], db, dd, ds;
    __asm__ volatile("v_cvt_u16_f16_e32 %0, %1\n\t" : "=v"(db) : "v"(x));
    __asm__ volatile("v_cvt_u16_f16_dpp %0, %1 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t" : "=v"(dd) : "v"(x));
    __asm__ volatile("v_cvt_u16_f16_sdwa %0, %1 dst_sel:WORD_0 "
                     "dst_unused:UNUSED_PAD src0_sel:WORD_0\n\t" : "=v"(ds) : "v"(x));
    o_base[i]=db; o_dpp[i]=dd; o_sdwa[i]=ds;
}

static uint16_t hbf(float v){ __half h=__float2half(v); uint16_t b; __builtin_memcpy(&b,&h,2); return b; }
static float   hf(uint16_t b){ __half h; __builtin_memcpy(&h,&b,2); return __half2float(h); }

int main(void)
{
    uint16_t *h_a=(uint16_t*)malloc(N*2);
    uint16_t *h_base=(uint16_t*)malloc(N*2),*h_dpp=(uint16_t*)malloc(N*2),*h_sdwa=(uint16_t*)malloc(N*2);
    for(int i=0;i<N;i++) h_a[i]=hbf((float)(i % 301));       /* exact int 0..300 */
    uint16_t *d_a,*d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_base,N*2));GPU_CHECK(hipMalloc(&d_dpp,N*2));GPU_CHECK(hipMalloc(&d_sdwa,N*2));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_base,d_dpp,d_sdwa,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base,d_base,N*2,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp,d_dpp,N*2,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa,d_sdwa,N*2,hipMemcpyDeviceToHost));
    const int perm[4]={1,0,3,2};
    int pass=1;
    for(int i=0;i<N && pass;i++){
        int lane=i%BLOCK_SIZE, base=i-lane, src=(lane&~3)|perm[lane&3];
        uint16_t eb=(uint16_t)hf(h_a[i]);
        uint16_t ed=(uint16_t)hf(h_a[base+src]);
        uint16_t es=(uint16_t)hf(h_a[i]);
        if(h_base[i]!=eb){fprintf(stderr,"FAIL base[%d] got %u exp %u\n",i,h_base[i],eb);pass=0;}
        else if(h_dpp[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %u exp %u\n",i,h_dpp[i],ed);pass=0;}
        else if(h_sdwa[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got %u exp %u\n",i,h_sdwa[i],es);pass=0;}
    }
    if(pass)printf("base[1]=%u dpp[1]=%u sdwa[1]=%u\n",h_base[1],h_dpp[1],h_sdwa[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_base));GPU_CHECK(hipFree(d_dpp));GPU_CHECK(hipFree(d_sdwa));
    free(h_a);free(h_base);free(h_dpp);free(h_sdwa);
    return pass?0:1;
}
