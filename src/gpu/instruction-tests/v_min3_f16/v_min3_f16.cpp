/*
 * HIP test for V_MIN3_F16 (ENC_VOP3, minimum of three f16).
 * gem5 (vop3.cc): D.f16 = fplibMin(fplibMin(S0,S1),S2) (IEEE minNum).
 *
 * NOTE on variants: gem5's execute() has panic_if(isDPPInst())/
 * panic_if(isSDWAInst()) guards, so only BASE is exercised (variants=base).
 *
 * Inputs are exactly-representable f16 (multiples of 0.25), no NaN/Inf, so the
 * result is bit-identical to one operand.  Pattern A (vector).
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const __half *a, const __half *b, const __half *c, __half *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint16_t ha,hb,hc;
    __builtin_memcpy(&ha,a+i,2);__builtin_memcpy(&hb,b+i,2);__builtin_memcpy(&hc,c+i,2);
    uint32_t s0=ha,s1=hb,s2=hc,d;
    __asm__ volatile("v_min3_f16 %0, %1, %2, %3\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1), "v"(s2));
    uint16_t hd=(uint16_t)(d&0xffff);
    __builtin_memcpy(out+i,&hd,2);
}

static uint16_t hb_(__half h){ uint16_t b; __builtin_memcpy(&b,&h,2); return b; }

int main(void)
{
    __half *h_a=(__half*)malloc(N*2),*h_b=(__half*)malloc(N*2);
    __half *h_c=(__half*)malloc(N*2),*h_out=(__half*)malloc(N*2);
    for (int i=0;i<N;i++){
        h_a[i]=__float2half((float)((i%23)-11)*0.25f);
        h_b[i]=__float2half((float)((i%17)-8)*0.5f);
        h_c[i]=__float2half((float)((i%13)-6)*0.75f);
    }
    __half *d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));
    GPU_CHECK(hipMalloc(&d_c,N*2));GPU_CHECK(hipMalloc(&d_out,N*2));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*2,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*2,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        float fa=__half2float(h_a[i]),fb=__half2float(h_b[i]),fc=__half2float(h_c[i]);
        float fm=fa<fb?fa:fb; fm=fm<fc?fm:fc;
        uint16_t exp=hb_(__float2half(fm));
        if(hb_(h_out[i])!=exp){fprintf(stderr,"FAIL[%d] got %04x exp %04x\n",i,hb_(h_out[i]),exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%.3f\n",__half2float(h_out[1]));
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_c));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_c);free(h_out);
    return pass?0:1;
}
