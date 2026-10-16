/*
 * HIP test for V_RNDNE_F16 (ENC_VOP1) plus its DPP and SDWA variants.
 * gem5 semantics: D.f16 = round_to_nearest_even(S0.f16).
 *
 * Three kernels exercise three encodings:
 *   base : v_rndne_f16_e32  vdst, vsrc
 *   dpp  : v_rndne_f16_dpp  vdst, vsrc quad_perm:[1,0,3,2]  (round the swizzled
 *          neighbor within each group of 4 lanes)
 *   sdwa : v_rndne_f16_sdwa vdst, vsrc dst_sel:WORD_0 dst_unused:UNUSED_PAD
 *          src0_sel:WORD_0  (identity sub-word select -> same as base)
 *
 * Inputs are multiples of 0.5 so the half-way cases exercise round-half-to-even
 * (matched on the host with rintf under the default rounding mode).  Results
 * are integer-valued f16 (exact).  Variants: base,dpp,sdwa
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
    uint16_t x=a[i], db, dd, ds;
    __asm__ volatile("v_rndne_f16_e32 %0, %1\n\t" : "=v"(db) : "v"(x));
    __asm__ volatile("v_rndne_f16_dpp %0, %1 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t" : "=v"(dd) : "v"(x));
    __asm__ volatile("v_rndne_f16_sdwa %0, %1 dst_sel:WORD_0 "
                     "dst_unused:UNUSED_PAD src0_sel:WORD_0\n\t" : "=v"(ds) : "v"(x));
    o_base[i]=db; o_dpp[i]=dd; o_sdwa[i]=ds;
}

static uint16_t hbf(float v){ __half h=__float2half(v); uint16_t b; __builtin_memcpy(&b,&h,2); return b; }
static float hf(uint16_t b){ __half h; __builtin_memcpy(&h,&b,2); return __half2float(h); }

int main(void)
{
    uint16_t *h_a=(uint16_t*)malloc(N*2);
    uint16_t *h_base=(uint16_t*)malloc(N*2),*h_dpp=(uint16_t*)malloc(N*2),*h_sdwa=(uint16_t*)malloc(N*2);
    for(int i=0;i<N;i++) h_a[i]=hbf((float)((i%81)-40)*0.5f);
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
        uint16_t eb=hbf(rintf(hf(h_a[i])));
        uint16_t ed=hbf(rintf(hf(h_a[base+src])));
        uint16_t es=hbf(rintf(hf(h_a[i])));
        if(h_base[i]!=eb){fprintf(stderr,"FAIL base[%d] got %04x exp %04x\n",i,h_base[i],eb);pass=0;}
        else if(h_dpp[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %04x exp %04x\n",i,h_dpp[i],ed);pass=0;}
        else if(h_sdwa[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got %04x exp %04x\n",i,h_sdwa[i],es);pass=0;}
    }
    if(pass)printf("base[1]=%04x dpp[1]=%04x sdwa[1]=%04x\n",h_base[1],h_dpp[1],h_sdwa[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_base));GPU_CHECK(hipFree(d_dpp));GPU_CHECK(hipFree(d_sdwa));
    free(h_a);free(h_base);free(h_dpp);free(h_sdwa);
    return pass?0:1;
}
