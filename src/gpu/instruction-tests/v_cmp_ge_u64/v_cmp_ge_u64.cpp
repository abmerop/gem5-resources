/*
 * HIP test for V_CMP_GE_U64 (VOPC/VOP3 compare, unsigned 64-bit >=).
 *   mask[lane] = (S0.u64 >= S1.u64). Pattern A, base only.
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cmp_ge_u64 v_cmp_ge_u64.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint64_t*a,const uint64_t*b,int*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t va=a[i],vb=b[i]; uint64_t mask;
    __asm__ volatile("v_cmp_ge_u64_e64 %0, %1, %2\n\t":"=s"(mask):"v"(va),"v"(vb));
    out[i]=(int)((mask>>threadIdx.x)&1ULL);
}
int main(void){
    uint64_t*h_a=(uint64_t*)malloc(N*8),*h_b=(uint64_t*)malloc(N*8); int*h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){h_a[i]=(uint64_t)i*0x9e3779b97f4a7c15ULL;h_b[i]=(uint64_t)(N-i)*0x100000001ULL; if(i%11==0)h_b[i]=h_a[i];}
    uint64_t*da,*db;int*dout;GPU_CHECK(hipMalloc(&da,N*8));GPU_CHECK(hipMalloc(&db,N*8));GPU_CHECK(hipMalloc(&dout,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(da,h_a,N*8,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,h_b,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dout,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,dout,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1; for(int i=0;i<N;i++){int exp=(h_a[i]>=h_b[i])?1:0; if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
