/*
 * HIP test for S_CMP_LE_I32 (SOPC: SCC = (S0.i32 <= S1.i32)).
 * gem5 (sopc.cc): scc = (src0.i32 <= src1.i32).  Captured via S_CSELECT_B32.
 * Pattern B (scalar).  Variant: base.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
__global__ void kern(int32_t a, int32_t b, uint32_t *out){
    uint32_t sdst;
    __asm__ volatile("s_cmp_le_i32 %1, %2\n\t"
                     "s_cselect_b32 %0, 1, 0\n\t"
                     : "=s"(sdst) : "s"(a), "s"(b) : "scc");
    if (threadIdx.x==0) out[blockIdx.x]=sdst;
}
int main(void){
    int32_t A[]={5,5,6,-1,-2147483648,2147483647,0,100};
    int32_t B[]={5,6,5,-1,2147483647,-2147483648,-1,100};
    const int NC=sizeof(A)/sizeof(A[0]);
    uint32_t *d_out; GPU_CHECK(hipMalloc(&d_out,NC*4));
    for(int c=0;c<NC;c++) kern<<<1,BLOCK_SIZE>>>(A[c],B[c],d_out+c);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    uint32_t h[64]; GPU_CHECK(hipMemcpy(h,d_out,NC*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int c=0;c<NC;c++){uint32_t exp=(A[c]<=B[c])?1u:0u;
        if(h[c]!=exp){fprintf(stderr,"FAIL[%d] %d<=%d got %u exp %u\n",c,A[c],B[c],h[c],exp);pass=0;}}
    if(pass)printf("out[0]=%u out[1]=%u\n",h[0],h[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_out));return pass?0:1;
}
