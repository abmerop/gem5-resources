/*
 * HIP test for V_DIV_FIXUP_LEGACY_F16 (ENC_VOP3), pattern A, base only.
 *
 * The legacy opcode decodes to Inst_VOP3__V_DIV_FIXUP_F16 in gem5's vega
 * decoder (gpu_decoder.cc), so this exercises the (fixed 2026-07-14,
 * commit b466f24) V_DIV_FIXUP_F16 execute path.
 *
 * Semantics (common/insts/vop3.cc), for finite non-zero, non-special S1/S2:
 *   sign_out = signbit(S1) ^ signbit(S2)
 *   D0       = |S0| with sign set to sign_out   (== copysign(|S0|, sign_out))
 * (Special cases handle NaN/Inf/0-over-0; this test stays in the normal path
 *  so the result is exactly f16(copysign(fabs(s0), sign(s1)^sign(s2))).)
 *
 * No abs/neg/omod/clamp modifiers -> result magnitude equals |s0| exactly
 * (f16-representable inputs), so the comparison is bit-exact.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const _Float16*a,const _Float16*b,const _Float16*c,
                       _Float16*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    _Float16 s0=a[i],s1=b[i],s2=c[i]; _Float16 d;
    __asm__ volatile("v_div_fixup_legacy_f16 %0, %1, %2, %3\n\t"
                     :"=v"(d):"v"(s0),"v"(s1),"v"(s2));
    out[i]=d;
}

int main(void){
    /* finite non-zero f16 magnitudes (exact in f16) + both signs */
    static const float mag[]={0.5f,1.0f,2.0f,0.25f,4.0f,1.5f,3.0f,0.125f};
    static const float sgn[]={1.0f,-1.0f};
    const int M=(int)(sizeof(mag)/sizeof(mag[0]));
    _Float16*h_a=(_Float16*)malloc(N*2),*h_b=(_Float16*)malloc(N*2),
            *h_c=(_Float16*)malloc(N*2),*h_out=(_Float16*)malloc(N*2);
    for(int i=0;i<N;i++){
        h_a[i]=(_Float16)(mag[i%M]*sgn[(i)&1]);       /* S0 (numerator/quotient) */
        h_b[i]=(_Float16)(mag[(i+3)%M]*sgn[(i>>1)&1]);/* S1 */
        h_c[i]=(_Float16)(mag[(i+5)%M]*sgn[(i>>2)&1]);/* S2 (denominator) */
    }
    _Float16*d_a,*d_b,*d_c,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));
    GPU_CHECK(hipMalloc(&d_c,N*2));GPU_CHECK(hipMalloc(&d_out,N*2));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*2,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*2,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        float s0=(float)h_a[i],s1=(float)h_b[i],s2=(float)h_c[i];
        int sign_out=(signbit(s1)!=0)^(signbit(s2)!=0);
        float ef=copysignf(fabsf(s0), sign_out?-1.0f:1.0f);
        _Float16 e=(_Float16)ef;
        uint16_t gb,eb; memcpy(&gb,&h_out[i],2); memcpy(&eb,&e,2);
        if(gb!=eb){fprintf(stderr,"FAIL[%d] got 0x%04x exp 0x%04x (s0=%g s1=%g s2=%g)\n",
                           i,gb,eb,s0,s1,s2);pass=0;break;}}
    if(pass)printf("out[0]=%g out[1]=%g\n",(float)h_out[0],(float)h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
