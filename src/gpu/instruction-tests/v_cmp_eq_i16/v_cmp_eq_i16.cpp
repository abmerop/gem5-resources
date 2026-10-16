/*
 * HIP test for V_CMP_EQ_I16 (VOPC/VOP3 compare, 16-bit integer equal).
 *   mask[lane] = (S0[15:0] == S1[15:0]). Pattern A, base only.
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cmp_eq_i16 v_cmp_eq_i16.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*a,const uint32_t*b,int*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t va=a[i],vb=b[i]; uint64_t mask;
    __asm__ volatile("v_cmp_eq_i16_e64 %0, %1, %2\n\t":"=s"(mask):"v"(va),"v"(vb));
    out[i]=(int)((mask>>threadIdx.x)&1ULL);
}
int main(void){
    uint32_t*h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4); int*h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){h_a[i]=(uint32_t)(uint16_t)(i%7);h_b[i]=(uint32_t)(uint16_t)(i%5);}
    uint32_t*da,*db;int*dout;GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dout,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(da,h_a,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,h_b,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dout,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,dout,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1; for(int i=0;i<N;i++){int exp=((uint16_t)h_a[i]==(uint16_t)h_b[i])?1:0; if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
