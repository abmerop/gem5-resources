/*
 * HIP test for V_CMP_TRU_F64 (VOPC/VOP3 compare, "TRU" = always true, f64).
 *   mask[lane] = 1 (unconditionally). Pattern A, base only.
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cmp_tru_f64 v_cmp_tru_f64.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const double*a,const double*b,int*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double va=a[i],vb=b[i]; uint64_t mask;
    __asm__ volatile("v_cmp_tru_f64_e64 %0, %1, %2\n\t":"=s"(mask):"v"(va),"v"(vb));
    out[i]=(int)((mask>>threadIdx.x)&1ULL);
}
int main(void){
    double*h_a=(double*)malloc(N*8),*h_b=(double*)malloc(N*8); int*h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){h_a[i]=(double)(i-128)*0.5; h_b[i]=(double)(i)*0.25; if(i%17==0)h_a[i]=nan("");}
    double*da,*db;int*dout;GPU_CHECK(hipMalloc(&da,N*8));GPU_CHECK(hipMalloc(&db,N*8));GPU_CHECK(hipMalloc(&dout,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(da,h_a,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,h_b,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dout,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,dout,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1; for(int i=0;i<N;i++){if(h_out[i]!=1){fprintf(stderr,"FAIL[%d] got %d exp 1\n",i,h_out[i]);pass=0;break;}}
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
