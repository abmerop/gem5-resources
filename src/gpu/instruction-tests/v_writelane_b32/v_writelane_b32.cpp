/*
 * HIP test for V_WRITELANE_B32 (VOP3, write a scalar into one lane of a VGPR).
 *
 * gem5 (Inst_VOP3__V_WRITELANE_B32::execute, vop3.cc):
 *   VDST[ SRC1 & 0x3f ] = SRC0     (only the selected lane is modified;
 *                                    all other lanes of VDST keep their value)
 * SRC0 is a scalar (wavefront-uniform) data value, SRC1 is the lane select.
 *
 * Test: one wavefront (64 lanes) per block.  Each lane seeds its VGPR with a
 * distinct per-lane value, then v_writelane_b32 overwrites lane LANE_SEL with a
 * uniform scalar WVAL.  Expected: lane LANE_SEL reads WVAL, every other lane
 * keeps its seed.  (SRC1 is an inline constant so the two SGPR sources don't
 * both hit the constant bus.)
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
#define LANE_SEL 17

__global__ void
kern(const uint32_t *in, uint32_t wval, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t v = in[i];              /* per-lane seed of VDST */
    __asm__ volatile("v_writelane_b32 %0, %1, %2\n\t"
                     : "+v"(v) : "s"(wval), "I"(LANE_SEL));
    out[i] = v;
}

int main(void)
{
    uint32_t *h_in=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    const uint32_t WVAL = 0xCAFEF00Du;
    for (int i=0;i<N;i++) h_in[i]=(uint32_t)(i*2654435761u+7u);
    uint32_t *d_in,*d_out;
    GPU_CHECK(hipMalloc(&d_in,N*4)); GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,WVAL,d_out,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++){
        int lane=i%BLOCK_SIZE;
        uint32_t exp = (lane==LANE_SEL) ? WVAL : h_in[i];
        if (h_out[i]!=exp){fprintf(stderr,"FAIL[%d] lane%d got 0x%08x exp 0x%08x\n",
            i,lane,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("lane%d=0x%08x other[0]=0x%08x\n",LANE_SEL,h_out[LANE_SEL],h_out[0]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_in)); GPU_CHECK(hipFree(d_out));
    free(h_in); free(h_out);
    return pass?0:1;
}
