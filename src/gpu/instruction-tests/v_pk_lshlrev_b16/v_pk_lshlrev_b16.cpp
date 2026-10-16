/*
 * HIP test for V_PK_LSHLREV_B16 (ENC_VOP3P, packed 16-bit reverse left shift).
 *
 *   D[15:0]  = S1[15:0]  << (S0[3:0])
 *   D[31:16] = S1[31:16] << (S0[19:16] & 0xf)
 *
 * "rev": shift amount is src0, value is src1.  gem5 uses bits(S0,3,0) of each
 * packed half as the shift.  Pattern A (vector), base only.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_pk_lshlrev_b16 v_pk_lshlrev_b16.cpp
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
kernel(const uint32_t *sh, const uint32_t *val, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t s0 = sh[i], s1 = val[i], vd;
    __asm__ volatile("v_pk_lshlrev_b16 %0, %1, %2\n\t" : "=v"(vd) : "v"(s0), "v"(s1));
    out[i] = vd;
}

int main(void)
{
    uint32_t *h_sh=(uint32_t*)malloc(N*4),*h_val=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){
        uint16_t sh0=(i)&0xf, sh1=(i>>2)&0xf;
        uint16_t v0=(uint16_t)(i*97+1), v1=(uint16_t)(i*333+7);
        h_sh[i]=((uint32_t)sh1<<16)|sh0;
        h_val[i]=((uint32_t)v1<<16)|v0;
    }
    uint32_t *d_sh,*d_val,*d_out;
    GPU_CHECK(hipMalloc(&d_sh,N*4));GPU_CHECK(hipMalloc(&d_val,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_sh,h_sh,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_val,h_val,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_sh,d_val,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint16_t s0lo=h_sh[i]&0xf, s0hi=(h_sh[i]>>16)&0xf;
        uint16_t v0=h_val[i]&0xffff, v1=h_val[i]>>16;
        uint16_t lo=(uint16_t)(v0<<s0lo), hi=(uint16_t)(v1<<s0hi);
        uint32_t exp=((uint32_t)hi<<16)|lo;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
