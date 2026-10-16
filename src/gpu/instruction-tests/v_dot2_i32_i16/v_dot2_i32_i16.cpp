/*
 * HIP test for V_DOT2_I32_I16 (ENC_VOP3P, 2-element i16 dot product + acc).
 *
 *   D.i32 = sext16(S0.lo)*sext16(S1.lo) + sext16(S0.hi)*sext16(S1.hi) + S2.i32
 *
 * Hardware accumulates the full-width products (no truncation of the
 * individual products).  Inputs are chosen so |product| > 2^15, which
 * exercises the full-width accumulation.  Pattern A, base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_dot2_i32_i16 v_dot2_i32_i16.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*a,const uint32_t*b,const int32_t*c,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t va=a[i],vb=b[i]; int32_t vc=c[i],vd;
    __asm__ volatile("v_dot2_i32_i16 %0, %1, %2, %3\n\t":"=v"(vd):"v"(va),"v"(vb),"v"(vc));
    out[i]=vd;
}
int main(void){
    uint32_t*h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    int32_t*h_c=(int32_t*)malloc(N*4),*h_out=(int32_t*)malloc(N*4);
    int16_t A0[N],A1[N],B0[N],B1[N];int32_t C[N];
    for(int i=0;i<N;i++){
        A0[i]=(int16_t)(((i%131)-65)*10); A1[i]=(int16_t)(((i%97)-48)*10);
        B0[i]=(int16_t)(((i%113)-56)*10); B1[i]=(int16_t)(((i%89)-44)*10);
        C[i]=(i*37)-5000;
        h_a[i]=((uint32_t)(uint16_t)A1[i]<<16)|(uint16_t)A0[i];
        h_b[i]=((uint32_t)(uint16_t)B1[i]<<16)|(uint16_t)B0[i];
        h_c[i]=C[i];
    }
    uint32_t*da,*db;int32_t*dc,*dout;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&dout,N*4));
    GPU_CHECK(hipMemcpy(da,h_a,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,h_b,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,h_c,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,dout,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,dout,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){int32_t exp=(int32_t)A0[i]*B0[i]+(int32_t)A1[i]*B1[i]+C[i];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
