/*
 * HIP test for V_ADD_I16 (v_add_i16, ENC_VOP3), CDNA4 / gfx950.
 * gem5 semantics: D.i16 = S0.i16 + S1.i16 (16-bit wrap).
 * Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N=256, BLOCK_SIZE=64;
__global__ void
kern(const int32_t *a, const int32_t *b, int32_t *o, int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint32_t va=(uint32_t)(a[i]&0xffff), vb=(uint32_t)(b[i]&0xffff), vd;
    __asm__ volatile("v_add_i16 %0, %1, %2\n\t":"=v"(vd):"v"(va),"v"(vb));
    o[i]=(int32_t)(int16_t)(vd & 0xffff);
}
int main(void)
{
    int32_t *h_a=(int32_t*)malloc(N*4),*h_b=(int32_t*)malloc(N*4),*h_o=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ h_a[i]=(int16_t)((i*131)%65536-32768); h_b[i]=(int16_t)((i*977+7)%65536-32768); }
    int32_t *d_a,*d_b,*d_o;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_o,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){
        int16_t exp=(int16_t)((int16_t)h_a[i] + (int16_t)h_b[i]);
        if((int16_t)h_o[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,(int)(int16_t)h_o[i],(int)exp);pass=0;}
    }
    if(pass)printf("o[1]=%d o[2]=%d\n",(int)(int16_t)h_o[1],(int)(int16_t)h_o[2]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_o));
    free(h_a);free(h_b);free(h_o); return pass?0:1;
}
