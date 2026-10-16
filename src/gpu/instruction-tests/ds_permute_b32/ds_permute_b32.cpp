/*
 * HIP test for DS_PERMUTE_B32 (ENC_DS, cross-lane forward permute / scatter).
 *
 * gem5 (ds.cc Inst_DS__DS_PERMUTE_B32::execute), per active lane:
 *   index       = bits(ADDR[lane] + OFFSET0, 7, 2)   // (byte addr >> 2) & 63
 *   VDST[index] = DATA[lane]                           // scatter to lane index
 * (lanes not targeted by any scatter read 0; here the mapping is a bijection,
 *  the mirror permutation, so every output lane is written exactly once.)
 *
 * Test: one wavefront (64 lanes) per block.  Lane L scatters its DATA to lane
 * (63 - L) via ADDR = (63 - L) * 4.  Result: out lane J receives DATA from lane
 * (63 - J), i.e. out[i] = data of the mirror lane.  Base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const uint32_t *in, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int lane = threadIdx.x;
    uint32_t data = in[i];
    uint32_t addr = (uint32_t)((BLOCK_SIZE - 1 - lane) * 4);
    uint32_t res;
    __asm__ volatile("ds_permute_b32 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t"
                     : "=v"(res) : "v"(addr), "v"(data));
    out[i] = res;
}

int main(void)
{
    uint32_t *h_in=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++) h_in[i]=(uint32_t)(i*40503u+0x1234u);
    uint32_t *d_in,*d_out;
    GPU_CHECK(hipMalloc(&d_in,N*4)); GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int wf_base=(i/BLOCK_SIZE)*BLOCK_SIZE, lane=i%BLOCK_SIZE;
        uint32_t exp=h_in[wf_base + (BLOCK_SIZE-1-lane)];
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[0]=0x%08x out[63]=0x%08x\n",h_out[0],h_out[63]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_in)); GPU_CHECK(hipFree(d_out));
    free(h_in); free(h_out);
    return pass?0:1;
}
