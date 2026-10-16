/*
 * HIP test for GLOBAL_LOAD_LDS_SSHORT (ENC_FLAT_GLBL, direct global->LDS DMA
 * load of a SIGNED 16-bit halfword, sign-extended into a 32-bit LDS dword).
 * Pattern C.
 *
 * Same DMA path as GLOBAL_LOAD_LDS_USHORT but the halfword is sign-extended
 * rather than zero-extended.  The hardware writes the sign-extended 16-bit
 * value into LDS at a per-lane dword slot (m0 supplies the LDS base offset).
 * There is no VGPR destination.  The 64-bit source address is the only
 * inline-asm value operand.  Inputs span >0x8000 so the sign-extension is
 * actually exercised (negative results distinguish it from the ushort form).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o global_load_lds_sshort \
 *        global_load_lds_sshort.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint16_t*g,int*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ int lds[BLOCK_SIZE];     /* only shared alloc -> LDS offset 0 */
    int t=threadIdx.x;
    uint64_t addr=(uint64_t)(g+i);
    uint32_t lds_off=(uint32_t)(uintptr_t)lds;
    __asm__ volatile(
        "s_mov_b32 m0, %1\n\t"
        "global_load_lds_sshort %0, off\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        :
        :"v"(addr),"s"(lds_off)
        :"memory","m0");
    __syncthreads();
    out[i]=lds[t];
}
int main(void){
    uint16_t*h_g=(uint16_t*)malloc(N*2); int*h_out=(int*)malloc(N*4);
    for(int i=0;i<N;i++) h_g[i]=(uint16_t)(i*517+3);   /* spans >0x8000 */
    uint16_t*d_g; int*d_out;
    GPU_CHECK(hipMalloc(&d_g,N*2));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_g,h_g,N*2,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_g,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){int exp=(int)(int16_t)h_g[i];   /* sign-extended */
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=%d out[%d]=%d\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
