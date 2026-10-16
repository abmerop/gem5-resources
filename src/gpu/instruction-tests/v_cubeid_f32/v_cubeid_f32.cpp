/*
 * HIP test for V_CUBEID_F32 (VOP3, cubemap face id selection).
 * gem5 (vop3.cc Inst_VOP3__V_CUBEID_F32) per lane, with S0=x, S1=y, S2=z:
 *   if |S2|>=|S0| && |S2|>=|S1|:  D = (S2<0)? 5 : 4
 *   elif |S1|>=|S0|:              D = (S1<0)? 3 : 2
 *   else:                         D = (S0<0)? 1 : 0
 * Inputs are chosen with distinct magnitudes so the >= ties never trigger;
 * exact float result (a small integer face id).  Base only.
 */
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const float*a,const float*b,const float*c,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i],s1=b[i],s2=c[i],d;
    __asm__ volatile("v_cubeid_f32 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0),"v"(s1),"v"(s2));
    out[i]=d;
}
static float ref(float s0,float s1,float s2){
    if(fabsf(s2)>=fabsf(s0)&&fabsf(s2)>=fabsf(s1)) return s2<0.0f?5.0f:4.0f;
    else if(fabsf(s1)>=fabsf(s0))                  return s1<0.0f?3.0f:2.0f;
    else                                           return s0<0.0f?1.0f:0.0f;
}
int main(void){
    float*h_a=(float*)malloc(N*4),*h_b=(float*)malloc(N*4);
    float*h_c=(float*)malloc(N*4),*h_out=(float*)malloc(N*4);
    for(int i=0;i<N;i++){
        /* distinct magnitudes; rotate which axis dominates + signs */
        h_a[i]=((i&1)?-1.0f:1.0f)*(float)(1+(i%7));
        h_b[i]=((i&2)?-1.0f:1.0f)*(float)(10+(i%5))*1.3f;
        h_c[i]=((i&4)?-1.0f:1.0f)*(float)(30+(i%3))*1.7f;
    }
    float*d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_c,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ float e=ref(h_a[i],h_b[i],h_c[i]);
        if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,h_out[i],e);pass=0;break;}}
    if(pass)printf("out[0]=%g out[%d]=%g\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
