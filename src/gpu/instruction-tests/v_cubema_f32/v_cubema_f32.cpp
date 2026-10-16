/*
 * HIP test for V_CUBEMA_F32 (ENC_VOP3, cubemap major-axis), pattern A.
 *
 * gem5 (vop3.cc) selects the operand with the largest magnitude (with a
 * specific tie-break order) and doubles it:
 *
 *   if (|s2| >= |s0| && |s2| >= |s1|)  D = s2 * 2.0
 *   else if (|s1| >= |s0|)             D = s1 * 2.0
 *   else                               D = s0 * 2.0
 *
 * The host reference reproduces this exact branch order.  Inputs are chosen
 * with distinct magnitudes and varied signs to exercise every branch.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cubema_f32 v_cubema_f32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const float*a,const float*b,const float*c,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i],s1=b[i],s2=c[i],d;
    __asm__ volatile("v_cubema_f32 %0, %1, %2, %3\n\t":"=v"(d):"v"(s0),"v"(s1),"v"(s2));
    out[i]=d;
}
static float ref(float s0,float s1,float s2){
    if(fabsf(s2)>=fabsf(s0)&&fabsf(s2)>=fabsf(s1)) return s2*2.0f;
    else if(fabsf(s1)>=fabsf(s0)) return s1*2.0f;
    else return s0*2.0f;
}
int main(void){
    float*h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4),*h_c=(float*)malloc(N*4),*h_out=(float*)malloc(N*4);
    for(int i=0;i<N;i++){
        /* rotate which operand has the largest magnitude */
        float big=(float)(i+10), mid=(float)((i%7)+2), sml=(float)((i%3)+1);
        int r=i%3;
        h_a[i]=((i&1)?-1:1)*(r==0?big:sml);
        h_b[i]=((i&2)?-1:1)*(r==1?big:mid);
        h_c[i]=((i&4)?-1:1)*(r==2?big:sml);
    }
    float*d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){float exp=ref(h_a[i],h_b[i],h_c[i]);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %g exp %g (s0=%g s1=%g s2=%g)\n",i,h_out[i],exp,h_a[i],h_b[i],h_c[i]);pass=0;break;}}
    if(pass)printf("out[0]=%g out[%d]=%g\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
