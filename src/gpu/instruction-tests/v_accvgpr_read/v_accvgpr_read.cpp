/*
 * HIP test for V_ACCVGPR_READ (V_ACCVGPR_READ_B32, VOP3P-MAI encoding):
 *   VDST.b32 = ACC[SRC].b32     (copy an accumulator VGPR to a vector VGPR)
 *
 * The "a" input constraint forces the compiler to place the operand in an
 * AGPR (emitting v_accvgpr_write), then inline asm reads it back into a VGPR
 * with v_accvgpr_read_b32.  The round-tripped value must equal the input.
 * Pattern A (vector), base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_accvgpr_read v_accvgpr_read.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kern(const int32_t *a, int32_t *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    int32_t v=a[i], d;
    /* "a"(v) -> compiler writes v into an AGPR; read it back to a VGPR. */
    __asm__ volatile("v_accvgpr_read_b32 %0, %1\n\t" : "=v"(d) : "a"(v));
    out[i]=d;
}
int main(void){
    int32_t *h_a=(int32_t*)malloc(N*4),*h_o=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h_a[i]=(int32_t)(i*2654435761u ^ (i<<3));
    int32_t *d_a,*d_o;
    GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_o,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++) if(h_o[i]!=h_a[i]){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_o[i],h_a[i]);pass=0;break;}
    if(pass)printf("out[1]=%d out[255]=%d\n",h_o[1],h_o[255]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_o));free(h_a);free(h_o);
    return pass?0:1;
}
