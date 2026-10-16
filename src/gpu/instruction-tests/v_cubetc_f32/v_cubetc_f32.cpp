/*
 * HIP test for V_CUBETC_F32 (VOP3, cubemap T coordinate).
 * gem5 (vop3.cc) branch, XYZ=(s0,s1,s2):
 *   if |s2|>=|s0| && |s2|>=|s1|: vdst = -s1
 *   elif |s1|>=|s0|:             vdst = (s1<0)? -s2 : s2
 *   else:                        vdst = -s1
 * Inputs use distinct magnitudes so no |.|== tie is hit; the result is an exact
 * copy/negate -> bit-exact compare.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;
__global__ void kern(const float*a,const float*b,const float*c,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    float s0=a[i],s1=b[i],s2=c[i],d;
    __asm__ volatile("v_cubetc_f32 %0, %1, %2, %3\n\t":"=v"(d):"v"(s0),"v"(s1),"v"(s2));
    out[i]=d;
}
static float ref(float s0,float s1,float s2){
    if(fabsf(s2)>=fabsf(s0) && fabsf(s2)>=fabsf(s1)) return -s1;
    else if(fabsf(s1)>=fabsf(s0)) return (s1<0.0f)?-s2:s2;
    else return -s1;
}
int main(void){
    float*ha=(float*)malloc(N*4),*hb=(float*)malloc(N*4),*hc=(float*)malloc(N*4),*ho=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ ha[i]=(float)((i%11)-5)+0.5f; hb[i]=(float)((i%7)-3)*3.0f+0.25f; hc[i]=(float)((i%5)-2)*7.0f+0.75f; }
    float*da,*db,*dc,*doo;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&doo,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,doo,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,doo,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ float e=ref(ha[i],hb[i],hc[i]); if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,ho[i],e);pass=0;break;}}
    if(pass)printf("out[0]=%g out[1]=%g\n",ho[0],ho[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
