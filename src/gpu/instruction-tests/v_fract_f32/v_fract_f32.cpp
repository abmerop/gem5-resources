/*
 * HIP test for V_FRACT_F32 (ENC_VOP1) plus DPP and SDWA variants, CDNA4 / gfx950.
 *   base : v_fract_f32_e32  vdst, vsrc
 *   dpp  : v_fract_f32_dpp  vdst, vsrc quad_perm:[1,0,3,2]
 *   sdwa : v_fract_f32_sdwa vdst, vsrc dst_sel:DWORD src0_sel:DWORD (identity)
 * Variants exercised: base,dpp,sdwa
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

static const int N=256, BLOCK_SIZE=64;
__global__ void
kern(const float *a, float *ob, float *od, float *os, int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float x=a[i]; float db,dd,ds;
    __asm__ volatile("v_fract_f32_e32 %0, %1\n\t":"=v"(db):"v"(x));
    __asm__ volatile("v_fract_f32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t":"=v"(dd):"v"(x));
    __asm__ volatile("v_fract_f32_sdwa %0, %1 dst_sel:DWORD src0_sel:DWORD\n\t":"=v"(ds):"v"(x));
    ob[i]=db; od[i]=dd; os[i]=ds;
}
int main(void)
{
    float *h_a=(float*)malloc(N*4);
    float *h_b=(float*)malloc(N*sizeof(float)),*h_d=(float*)malloc(N*sizeof(float)),*h_s=(float*)malloc(N*sizeof(float));
    for(int i=0;i<N;i++) h_a[i]=(float)((i*7)%201-100) + ((i%4)*0.25f);
    float *d_a; float *d_b,*d_d,*d_s;
    GPU_CHECK(hipMalloc(&d_a,N*4));
    GPU_CHECK(hipMalloc(&d_b,N*sizeof(float)));GPU_CHECK(hipMalloc(&d_d,N*sizeof(float)));GPU_CHECK(hipMalloc(&d_s,N*sizeof(float)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_d,d_s,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_b,d_b,N*sizeof(float),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_d,N*sizeof(float),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_s,d_s,N*sizeof(float),hipMemcpyDeviceToHost));
    const int perm[4]={1,0,3,2};
    int pass=1;
    for(int i=0;i<N&&pass;i++){
        int lane=i%BLOCK_SIZE, base=i-lane, src=(lane&~3)|perm[lane&3];
        float xb=h_a[i], xd=h_a[base+src];
        // V_FRACT_F32 is defined as S - floor(S) (result in [0,1)), not
        // S - trunc(S); these differ for negative inputs.
        float eb=(xb-floorf(xb)), ed=(xd-floorf(xd)), es=(xb-floorf(xb));
        if(!(fabsf(h_b[i]-eb)<1e-6f)){fprintf(stderr,"FAIL base[%d] got %f exp %f\n",i,h_b[i],eb);pass=0;}
        else if(!(fabsf(h_d[i]-ed)<1e-6f)){fprintf(stderr,"FAIL dpp[%d] got %f exp %f\n",i,h_d[i],ed);pass=0;}
        else if(!(fabsf(h_s[i]-es)<1e-6f)){fprintf(stderr,"FAIL sdwa[%d] got %f exp %f\n",i,h_s[i],es);pass=0;}
    }
    if(pass)printf("base[1]=%f dpp[1]=%f sdwa[1]=%f\n",h_b[1],h_d[1],h_s[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_d));GPU_CHECK(hipFree(d_s));
    free(h_a);free(h_b);free(h_d);free(h_s); return pass?0:1;
}
