/*
 * HIP test for V_DOT2_F32_F16 (ENC_VOP3P, 2-element f16 dot to f32 + acc).
 *   D.f32 = (f32)(S0.lo*S1.lo) + (f32)(S0.hi*S1.hi) + S2.f32
 * gem5 computes each half product, converts to f32, sums in f32, then +S2.
 * All halves are tiny integers so every product/sum is exact in both half and
 * f32, making the result bit-checkable.  Pattern A, base only.
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_dot2_f32_f16 v_dot2_f32_f16.cpp
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*a,const uint32_t*b,const float*c,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t va=a[i],vb=b[i]; float vc=c[i],vd;
    __asm__ volatile("v_dot2_f32_f16 %0, %1, %2, %3\n\t":"=v"(vd):"v"(va),"v"(vb),"v"(vc));
    out[i]=vd;
}
static uint16_t hb(__half h){uint16_t b;__builtin_memcpy(&b,&h,2);return b;}
int main(void){
    uint32_t*h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4); float*h_c=(float*)malloc(N*4),*h_out=(float*)malloc(N*4);
    int A0[N],A1[N],B0[N],B1[N];float C[N];
    for(int i=0;i<N;i++){A0[i]=(i%7)-3;A1[i]=(i%5)-2;B0[i]=(i%6)-2;B1[i]=(i%9)-4;C[i]=(float)((i%21)-10);
        h_a[i]=((uint32_t)hb(__float2half((float)A1[i]))<<16)|hb(__float2half((float)A0[i]));
        h_b[i]=((uint32_t)hb(__float2half((float)B1[i]))<<16)|hb(__float2half((float)B0[i]));
        h_c[i]=C[i];}
    uint32_t*da,*db;float*dc,*dout;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&dout,N*4));
    GPU_CHECK(hipMemcpy(da,h_a,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,h_b,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,h_c,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,dout,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,dout,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){float exp=(float)(A0[i]*B0[i])+(float)(A1[i]*B1[i])+C[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %f exp %f\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%f out[%d]=%f\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
