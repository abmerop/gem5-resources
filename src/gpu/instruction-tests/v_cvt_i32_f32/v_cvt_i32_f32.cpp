/*
 * HIP test for V_CVT_I32_F32 (ENC_VOP1) plus its DPP and SDWA variants.
 * gem5 semantics: D.i32 = (int32)trunc(S0.f32)  (round toward zero).
 *
 * Three kernels exercise three encodings:
 *   base : v_cvt_i32_f32_e32  vdst, vsrc
 *   dpp  : v_cvt_i32_f32_dpp  vdst, vsrc quad_perm:[1,0,3,2]  (convert the
 *          swizzled neighbor's f32)
 *   sdwa : v_cvt_i32_f32_sdwa vdst, vsrc dst_sel:DWORD src0_sel:DWORD
 *          (identity sub-dword selects -> same result as base)
 *
 * Inputs are bounded non-integer f32 values -> truncation is well defined and
 * exact.  Variants exercised: base,dpp,sdwa
 */
#include <hip/hip_runtime.h>
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
kern(const float *a, int32_t *o_base, int32_t *o_dpp, int32_t *o_sdwa, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    int32_t db, dd, ds;
    __asm__ volatile("v_cvt_i32_f32_e32 %0, %1\n\t" : "=v"(db) : "v"(x));
    __asm__ volatile("v_cvt_i32_f32_dpp %0, %1 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t" : "=v"(dd) : "v"(x));
    __asm__ volatile("v_cvt_i32_f32_sdwa %0, %1 dst_sel:DWORD src0_sel:DWORD\n\t"
                     : "=v"(ds) : "v"(x));
    o_base[i]=db; o_dpp[i]=dd; o_sdwa[i]=ds;
}

int main(void)
{
    float *h_a=(float*)malloc(N*4);
    int32_t *h_b=(int32_t*)malloc(N*4),*h_d=(int32_t*)malloc(N*4),*h_s=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_a[i]=(float)((i*7)%201-100) + ((i%3)-1)*0.6f;
    float *d_a; int32_t *d_b,*d_d,*d_s;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_d,N*4));GPU_CHECK(hipMalloc(&d_s,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_d,d_s,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_b,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_d,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_s,d_s,N*4,hipMemcpyDeviceToHost));
    const int perm[4]={1,0,3,2};
    int pass=1;
    for(int i=0;i<N && pass;i++){
        int lane=i%BLOCK_SIZE, base=i-lane, src=(lane&~3)|perm[lane&3];
        int32_t eb=(int32_t)truncf(h_a[i]);
        int32_t ed=(int32_t)truncf(h_a[base+src]);
        int32_t es=(int32_t)truncf(h_a[i]);
        if(h_b[i]!=eb){fprintf(stderr,"FAIL base[%d] got %d exp %d\n",i,h_b[i],eb);pass=0;}
        else if(h_d[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %d exp %d\n",i,h_d[i],ed);pass=0;}
        else if(h_s[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got %d exp %d\n",i,h_s[i],es);pass=0;}
    }
    if(pass)printf("base[1]=%d dpp[1]=%d sdwa[1]=%d\n",h_b[1],h_d[1],h_s[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_d));GPU_CHECK(hipFree(d_s));
    free(h_a);free(h_b);free(h_d);free(h_s);
    return pass?0:1;
}
