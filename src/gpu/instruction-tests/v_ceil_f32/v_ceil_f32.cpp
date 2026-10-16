/*
 * HIP test for V_CEIL_F32 (ENC_VOP1) plus its DPP and SDWA variants.
 * gem5 semantics: D.f32 = ceil(S0.f32)  (round toward +inf to integer).
 *
 * Three kernels exercise three encodings of the same opcode:
 *   base : v_ceil_f32_e32  vdst, vsrc
 *   dpp  : v_ceil_f32_dpp  vdst, vsrc quad_perm:[1,0,3,2]  (swap lanes in pairs
 *          within each group of 4) -> ceil is applied to the swizzled neighbor
 *   sdwa : v_ceil_f32_sdwa vdst, vsrc dst_sel:DWORD src0_sel:DWORD  (identity
 *          sub-dword selects -> same result as base, but SDWA encoding)
 *
 * ceil of a finite f32 is exact -> bit-exact comparison is valid.
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

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const float *a, float *o_base, float *o_dpp, float *o_sdwa, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i], db, dd, ds;
    __asm__ volatile("v_ceil_f32_e32 %0, %1\n\t" : "=v"(db) : "v"(x));
    __asm__ volatile("v_ceil_f32_dpp %0, %1 quad_perm:[1,0,3,2] "
                     "row_mask:0xf bank_mask:0xf\n\t" : "=v"(dd) : "v"(x));
    __asm__ volatile("v_ceil_f32_sdwa %0, %1 dst_sel:DWORD src0_sel:DWORD\n\t"
                     : "=v"(ds) : "v"(x));
    o_base[i]=db; o_dpp[i]=dd; o_sdwa[i]=ds;
}

int main(void)
{
    float *h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4),*h_d=(float*)malloc(N*4),*h_s=(float*)malloc(N*4);
    for(int i=0;i<N;i++) h_a[i]=(float)((i*7)%101-50) + ((i%3)-1)*0.4f;
    float *d_a,*d_b,*d_d,*d_s;
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
        int lane=i%BLOCK_SIZE, base=i-lane;
        int src=(lane&~3)|perm[lane&3];
        float eb=ceilf(h_a[i]);
        float ed=ceilf(h_a[base+src]);
        float es=ceilf(h_a[i]);
        if(h_b[i]!=eb){fprintf(stderr,"FAIL base[%d] got %f exp %f\n",i,h_b[i],eb);pass=0;}
        else if(h_d[i]!=ed){fprintf(stderr,"FAIL dpp[%d] got %f exp %f\n",i,h_d[i],ed);pass=0;}
        else if(h_s[i]!=es){fprintf(stderr,"FAIL sdwa[%d] got %f exp %f\n",i,h_s[i],es);pass=0;}
    }
    if(pass)printf("base[1]=%f dpp[1]=%f sdwa[1]=%f\n",h_b[1],h_d[1],h_s[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_d));GPU_CHECK(hipFree(d_s));
    free(h_a);free(h_b);free(h_d);free(h_s);
    return pass?0:1;
}
