/*
 * HIP test for V_CVT_PKNORM_I16_F32 (VOP3).
 * Signed-normalized conversion: n = clamp(src, -1, 1) * 32767.0f (0x7FFF),
 * rounded to nearest even; vdst[15:0]=int16_t(n0), [31:16]=int16_t(n1).
 * Base only (op panics on DPP/SDWA/OPSEL).
 */
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static inline int16_t snorm16(float s){
    if(s> 1.0f)s= 1.0f;
    if(s<-1.0f)s=-1.0f;
    return (int16_t)rintf(s*32767.0f);  // round to nearest even
}
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const float*a,const float*b,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i], s1=b[i]; uint32_t d;
    __asm__ volatile("v_cvt_pknorm_i16_f32 %0, %1, %2\n\t":"=v"(d):"v"(s0),"v"(s1));
    out[i]=d;
}
int main(void){
    float*ha=(float*)malloc(N*4),*hb=(float*)malloc(N*4); uint32_t*ho=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ ha[i]=(float)((i%64)-32)/128.0f; hb[i]=(float)(((i+13)%64)-32)/128.0f; }
    float*da,*db; uint32_t*doo;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint16_t v0=(uint16_t)snorm16(ha[i]);
        uint16_t v1=(uint16_t)snorm16(hb[i]);
        uint32_t e=((uint32_t)v1<<16)|v0;
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,ho[i],e);pass=0;break;}}
    if(pass)printf("out[1]=%08x\n",ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
