/*
 * HIP test for V_CVT_SCALEF32_SR_FP8_F16 (ENC_VOP3, scaled stochastic-round
 * convert f16 -> fp8 E4M3), pattern A.
 *
 * gem5 (vop3_cvt.hh, Inst_VOP3__V_CVT_SCALEF32_SR):
 *   in    = mxfloat16(src0)            ; f16 input (raw bits in a u32)
 *   seed  = src1                       ; stochastic-round seed
 *   scale = src2.f32                   ; in.scaleDiv(scale)
 *   cvt   = convertMXFP<fp8_e4m3>(in/scale, roundStochastic, seed)
 *   vdst[byte OPSEL[3:2]] = fp8 byte
 *
 * Stochastic rounding is seed-dependent in general.  To make the result
 * deterministic and self-checkable, we choose inputs that are EXACTLY
 * representable in fp8 E4M3 after scaling (scale = 1.0).  When the scaled
 * value is exactly representable, both rounding candidates coincide and the
 * result is exact regardless of the seed.
 *
 * OPSEL defaults to 0 -> the fp8 byte is written to byte 0 of the destination
 * dword.  We initialise the destination to 0 and read back byte 0.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_scalef32_sr_fp8_f16 \
 *        v_cvt_scalef32_sr_fp8_f16.cpp
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const uint32_t*in_f16,const int32_t*seed,const float*scale,
                       uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t s0=in_f16[i]; int32_t s1=seed[i]; float s2=scale[i];
    uint32_t d=0;
    __asm__ volatile("v_cvt_scalef32_sr_fp8_f16 %0, %1, %2, %3\n\t"
        :"+v"(d):"v"(s0),"v"(s1),"v"(s2));
    out[i]=d;
}

/* OCP fp8 E4M3 encoder for positive, exactly-representable values. */
static uint8_t f32_to_e4m3(float v){
    if(v==0.0f) return 0;
    int sign=v<0?1:0; v=fabsf(v);
    int e; float m=frexpf(v,&e);      /* v = m*2^e, m in [0.5,1) */
    (void)m;
    int E=e-1;                        /* unbiased exponent of 1.xxx form */
    int exp_field=E+7;                /* bias 7 */
    float frac=v/ldexpf(1.0f,E)-1.0f; /* mantissa fraction in [0,1) */
    int man=(int)lroundf(frac*8.0f);  /* 3 mantissa bits */
    return (uint8_t)((sign<<7)|((exp_field&0xF)<<3)|(man&0x7));
}

int main(void){
    /* exactly representable in E4M3 */
    const float vals[8]={0.5f,1.0f,1.5f,2.0f,3.0f,4.0f,6.0f,8.0f};
    uint32_t*h_in=(uint32_t*)malloc(N*4); int32_t*h_seed=(int32_t*)malloc(N*4);
    float*h_scale=(float*)malloc(N*4); uint32_t*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){
        float v=vals[i%8];
        __half h=__float2half(v); uint16_t hb; memcpy(&hb,&h,2);
        h_in[i]=(uint32_t)hb;
        h_seed[i]=(int32_t)(i*2654435761u);   /* arbitrary seed */
        h_scale[i]=1.0f;
    }
    uint32_t*d_in,*d_out; int32_t*d_seed; float*d_scale;
    GPU_CHECK(hipMalloc(&d_in,N*4));GPU_CHECK(hipMalloc(&d_seed,N*4));
    GPU_CHECK(hipMalloc(&d_scale,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_in,h_in,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_seed,h_seed,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_scale,h_scale,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_in,d_seed,d_scale,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint8_t got=(uint8_t)(h_out[i]&0xFF);
        uint8_t exp=f32_to_e4m3(vals[i%8]);
        if(got!=exp){fprintf(stderr,"FAIL[%d] v=%g got 0x%02x exp 0x%02x\n",i,vals[i%8],got,exp);pass=0;break;}
    }
    if(pass)printf("out[0]=0x%02x out[%d]=0x%02x\n",h_out[0]&0xFF,N-1,h_out[N-1]&0xFF);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
