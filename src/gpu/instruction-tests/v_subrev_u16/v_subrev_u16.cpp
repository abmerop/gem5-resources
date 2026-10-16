/*
 * HIP test for V_SUBREV_U16 (ENC_VOP2) plus its DPP and SDWA variants.
 * gem5 semantics: D.u16 = S1.u16 - S0.u16  (reverse subtract, wraps mod 2^16).
 *
 * Three kernels exercise three encodings:
 *   base : v_subrev_u16_e32  vdst, vsrc0, vsrc1   -> s1 - s0
 *   dpp  : v_subrev_u16_dpp  vdst, vsrc0, vsrc1 quad_perm:[1,0,3,2]
 *          (on hardware the swizzle applies to the minuend s1) -> swizzled(s1) - s0
 *   sdwa : v_subrev_u16_sdwa vdst, vsrc0, vsrc1 dst_sel:WORD_0
 *          dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0 -> s1 - s0
 *
 * 16-bit unsigned arithmetic -> exact, compare low 16 bits.
 * Variants exercised: base,dpp,sdwa
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const uint16_t *a, const uint16_t *b,
     uint16_t *o_base, uint16_t *o_dpp, uint16_t *o_sdwa, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t s0=a[i], s1=b[i], db, dd, ds;
    __asm__ volatile("v_subrev_u16_e32 %0, %1, %2\n\t"
                     : "=v"(db) : "v"(s0), "v"(s1));
    __asm__ volatile("v_subrev_u16_dpp %0, %1, %2 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t"
                     : "=v"(dd) : "v"(s0), "v"(s1));
    __asm__ volatile("v_subrev_u16_sdwa %0, %1, %2 dst_sel:WORD_0 "
                     "dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_0\n\t"
                     : "=v"(ds) : "v"(s0), "v"(s1));
    o_base[i]=db; o_dpp[i]=dd; o_sdwa[i]=ds;
}

int main(void)
{
    uint16_t *h_a=(uint16_t*)malloc(N*2),*h_b=(uint16_t*)malloc(N*2);
    uint16_t *h_base=(uint16_t*)malloc(N*2),*h_dpp=(uint16_t*)malloc(N*2),*h_sdwa=(uint16_t*)malloc(N*2);
    for(int i=0;i<N;i++){ h_a[i]=(uint16_t)(i*257+13); h_b[i]=(uint16_t)(i*131+9999); }
    uint16_t *d_a,*d_b,*d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));
    GPU_CHECK(hipMalloc(&d_base,N*2));GPU_CHECK(hipMalloc(&d_dpp,N*2));GPU_CHECK(hipMalloc(&d_sdwa,N*2));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_base,d_dpp,d_sdwa,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base,d_base,N*2,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp,d_dpp,N*2,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa,d_sdwa,N*2,hipMemcpyDeviceToHost));
    const int perm[4]={1,0,3,2};
    int pass=1;
    for(int i=0;i<N && pass;i++){
        int lane=i%BLOCK_SIZE, base=i-lane, src=(lane&~3)|perm[lane&3];
        uint16_t eb=(uint16_t)(h_b[i]-h_a[i]);
        // On hardware, V_SUBREV's DPP lane swizzle applies to the minuend
        // (b, the first explicit source / SRC1), so the result is
        // b[swizzled] - a[i].
        uint16_t ed=(uint16_t)(h_b[base+src]-h_a[i]);
        uint16_t es=(uint16_t)(h_b[i]-h_a[i]);
        if(h_base[i]!=eb){fprintf(stderr,"FAIL base[%d] got %04x exp %04x\n",i,h_base[i],eb);pass=0;}
        else if(h_dpp[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %04x exp %04x\n",i,h_dpp[i],ed);pass=0;}
        else if(h_sdwa[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got %04x exp %04x\n",i,h_sdwa[i],es);pass=0;}
    }
    if(pass)printf("base[1]=%04x dpp[1]=%04x sdwa[1]=%04x\n",h_base[1],h_dpp[1],h_sdwa[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_base));GPU_CHECK(hipFree(d_dpp));GPU_CHECK(hipFree(d_sdwa));
    free(h_a);free(h_b);free(h_base);free(h_dpp);free(h_sdwa);
    return pass?0:1;
}
