/*
 * HIP test for V_SAT_PK_U8_I16 (ENC_VOP1) plus DPP and SDWA variants.
 *
 * gem5 semantics (vop1.cc): the 32-bit source packs two signed 16-bit ints
 *   i0 = src[15:0], i1 = src[31:16]
 *   D[7:0]   = sat8(i0)        sat8(x) = x<0 ? 0 : x>255 ? 255 : x
 *   D[15:8]  = sat8(i1)
 *   D[31:16] = 0
 *
 * Variants exercised:
 *   base : v_sat_pk_u8_i16_e32  vd, va
 *   dpp  : v_sat_pk_u8_i16_dpp  ... quad_perm:[1,0,3,2] (swizzled 32-bit src)
 *   sdwa : v_sat_pk_u8_i16_sdwa ... src0_sel:DWORD dst_sel:DWORD (identity)
 *
 * Variants: base,dpp,sdwa
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
kern(const uint32_t *a, uint32_t *o_base, uint32_t *o_dpp, uint32_t *o_sdwa, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x = a[i], db, dd, ds;
    __asm__ volatile("v_sat_pk_u8_i16_e32 %0, %1\n\t" : "=v"(db) : "v"(x));
    __asm__ volatile("v_sat_pk_u8_i16_dpp %0, %1 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t" : "=v"(dd) : "v"(x));
    __asm__ volatile("v_sat_pk_u8_i16_sdwa %0, %1 dst_sel:DWORD "
                     "src0_sel:DWORD\n\t" : "=v"(ds) : "v"(x));
    o_base[i]=db; o_dpp[i]=dd; o_sdwa[i]=ds;
}

static uint8_t sat8(int16_t x){ if(x<0) return 0; if(x>255) return 255; return (uint8_t)x; }
static uint32_t ref(uint32_t v){
    int16_t i0=(int16_t)(v & 0xffff), i1=(int16_t)(v >> 16);
    return (uint32_t)sat8(i0) | ((uint32_t)sat8(i1) << 8);
}

int main(void)
{
    uint32_t *h_a=(uint32_t*)malloc(N*4);
    uint32_t *h_base=(uint32_t*)malloc(N*4),*h_dpp=(uint32_t*)malloc(N*4),*h_sdwa=(uint32_t*)malloc(N*4);
    /* mix of negative, in-range, and >255 int16 halves */
    for(int i=0;i<N;i++){
        int16_t i0=(int16_t)((i*7) % 400 - 100);    /* -100..299 */
        int16_t i1=(int16_t)((i*13) % 700 - 300);   /* -300..399 */
        h_a[i]=((uint32_t)(uint16_t)i1<<16)|(uint16_t)i0;
    }
    uint32_t *d_a,*d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_base,N*4));GPU_CHECK(hipMalloc(&d_dpp,N*4));GPU_CHECK(hipMalloc(&d_sdwa,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_base,d_dpp,d_sdwa,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_base,d_base,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_dpp,d_dpp,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_sdwa,d_sdwa,N*4,hipMemcpyDeviceToHost));
    const int perm[4]={1,0,3,2};
    int pass=1;
    for(int i=0;i<N && pass;i++){
        int lane=i%BLOCK_SIZE, base=i-lane, src=(lane&~3)|perm[lane&3];
        uint32_t eb=ref(h_a[i]);
        uint32_t ed=ref(h_a[base+src]);
        uint32_t es=ref(h_a[i]);
        if(h_base[i]!=eb){fprintf(stderr,"FAIL base[%d] got %08x exp %08x\n",i,h_base[i],eb);pass=0;}
        else if(h_dpp[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %08x exp %08x\n",i,h_dpp[i],ed);pass=0;}
        else if(h_sdwa[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got %08x exp %08x\n",i,h_sdwa[i],es);pass=0;}
    }
    if(pass)printf("base[1]=%08x dpp[1]=%08x sdwa[1]=%08x\n",h_base[1],h_dpp[1],h_sdwa[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_base));GPU_CHECK(hipFree(d_dpp));GPU_CHECK(hipFree(d_sdwa));
    free(h_a);free(h_base);free(h_dpp);free(h_sdwa);
    return pass?0:1;
}
