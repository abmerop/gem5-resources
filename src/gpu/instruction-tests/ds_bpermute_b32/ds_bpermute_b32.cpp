/*
 * HIP test for DS_BPERMUTE_B32 (ENC_DS, cross-lane backward permute / gather).
 *
 * gem5 (ds.cc Inst_DS__DS_BPERMUTE_B32::execute), per active lane:
 *   index      = bits(ADDR[lane] + OFFSET0, 7, 2)   // (byte addr >> 2) & 63
 *   VDST[lane] = DATA[index]                          // gather from lane index
 * (if the source lane is inactive, 0 is written; here all 64 lanes are active).
 *
 * Test: one wavefront (64 lanes) per block.  Lane L reads DATA from lane
 * (63 - L) by setting ADDR = (63 - L) * 4.  Expected out[i] = data of the
 * mirror lane within the same wavefront.  Base only.
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
    __asm__ volatile("ds_bpermute_b32 %0, %1, %2\n\ts_waitcnt lgkmcnt(0)\n\t"
                     : "=v"(res) : "v"(addr), "v"(data));
    out[i] = res;
}

int main(void)
{
    uint32_t *h_in=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for (int i=0;i<N;i++) h_in[i]=(uint32_t)(i*2654435761u+99u);
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
