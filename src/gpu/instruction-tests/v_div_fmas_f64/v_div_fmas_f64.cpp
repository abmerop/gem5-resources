/*
 * HIP test for V_DIV_FMAS_F64 (VOP3): vdst = VCC ? 2^64*fma(a,b,c) : fma(a,b,c)
 * (gem5 vop3.cc; vdst correctly declared VecOperandF64).  We force VCC=0 for the
 * whole wave (s_mov_b64 vcc,0) so the result is the plain double fma(a,b,c).
 * Small exact integer operands -> bit-exact against fma.  Variants: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const double*a,const double*b,const double*c,double*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s0=a[i],s1=b[i],s2=c[i],r;
    __asm__ volatile("s_mov_b64 vcc, 0\n\tv_div_fmas_f64 %0, %1, %2, %3\n\t"
                     :"=v"(r):"v"(s0),"v"(s1),"v"(s2):"vcc");
    out[i]=r;
}
int main(void){
    double*ha=(double*)malloc(N*8),*hb=(double*)malloc(N*8),*hc=(double*)malloc(N*8),*ho=(double*)malloc(N*8);
    for(int i=0;i<N;i++){ha[i]=(double)(i%7)+1;hb[i]=(double)(i%5)+2;hc[i]=(double)(i%11);}
    double*da,*db,*dc,*dO;
    GPU_CHECK(hipMalloc(&da,N*8));GPU_CHECK(hipMalloc(&db,N*8));GPU_CHECK(hipMalloc(&dc,N*8));GPU_CHECK(hipMalloc(&dO,N*8));
    GPU_CHECK(hipMemcpy(da,ha,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){double e=fma(ha[i],hb[i],hc[i]);
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %g exp %g\n",i,ho[i],e);pass=0;break;}}
    if(pass)printf("out[0]=%g\n",ho[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
