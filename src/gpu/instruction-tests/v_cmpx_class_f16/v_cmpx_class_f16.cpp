/*
 * HIP test for V_CMPX_CLASS_F16 (v_cmpx_class_f16_e64), CDNA4 / gfx950.
 * V_CMP_CLASS: result = (fp16-class(S0) intersects the class bitmask S1) ? 1:0,
 * written per-lane into EXEC (and the SGPR dest of the e64 form).
 * Class bits (same 10-bit encoding as f32/f64):
 *   0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -subnormal, 5 -zero,
 *   6 +zero, 7 +subnormal, 8 +normal, 9 +inf.
 * The host reference classifies directly from the fp16 bit pattern so the
 * normal/subnormal boundary matches half precision (not float).
 * Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){\
 fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

static int cls_match_f16(uint16_t h, uint32_t m){
    int sign =(h>>15)&1;
    int exp  =(h>>10)&0x1f;
    int mant = h&0x3ff;
    int is_zero=(exp==0 && mant==0);
    int is_sub =(exp==0 && mant!=0);
    int is_inf =(exp==0x1f && mant==0);
    int is_nan =(exp==0x1f && mant!=0);
    int is_norm=!(exp==0 || exp==0x1f);
    if((m&3u) && is_nan) return 1;                 /* sNaN|qNaN */
    if((m&(1u<<2)) && is_inf && sign) return 1;    /* -inf */
    if((m&(1u<<3)) && is_norm && sign) return 1;   /* -normal */
    if((m&(1u<<4)) && is_sub && sign) return 1;    /* -subnormal */
    if((m&(1u<<5)) && is_zero && sign) return 1;   /* -zero */
    if((m&(1u<<6)) && is_zero && !sign) return 1;  /* +zero */
    if((m&(1u<<7)) && is_sub && !sign) return 1;   /* +subnormal */
    if((m&(1u<<8)) && is_norm && !sign) return 1;  /* +normal */
    if((m&(1u<<9)) && is_inf && !sign) return 1;   /* +inf */
    return 0;
}

/*
 * kernel_test: reads a fp16 value (raw bits in the low half of a VGPR) and a
 * per-lane class mask, runs v_cmpx_class_f16_e64 into an SGPR pair, saving and
 * restoring EXEC so the divergent write does not disturb control flow, then
 * extracts this lane's result bit.
 */
__global__ void
kernel_test(const uint16_t *a, const uint32_t *mask_in, int *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint32_t x=a[i];              /* fp16 bits, zero-extended into a VGPR */
    uint32_t m=mask_in[i];
    uint64_t mask, saved;
    __asm__ volatile(
        "s_mov_b64 %1, exec\n\t"
        "v_cmpx_class_f16_e64 %0, %2, %3\n\t"
        "s_mov_b64 exec, %1\n\t"
        : "=s"(mask), "=s"(saved) : "v"(x), "v"(m));
    out[i]=(int)((mask>>threadIdx.x)&1ULL);
}

int main(void){
    uint16_t *h_a=(uint16_t*)malloc(N*sizeof(uint16_t));
    uint32_t *h_m=(uint32_t*)malloc(N*sizeof(uint32_t));
    int *h_o=(int*)malloc(N*sizeof(int));
    /* fp16 representatives of each class */
    uint16_t vals[8]={
        0x4200,  /* +3.0  +normal    */
        0xC200,  /* -3.0  -normal    */
        0x0000,  /* +0.0  +zero      */
        0x8000,  /* -0.0  -zero      */
        0x7C00,  /* +inf             */
        0xFC00,  /* -inf             */
        0x7E00,  /* qNaN             */
        0x0001,  /* +subnormal       */
    };
    uint32_t masks[4]={0x3FFu, 0x040u|0x020u, 0x200u|0x004u, 0x100u|0x008u};
    for(int i=0;i<N;i++){ h_a[i]=vals[i%8]; h_m[i]=masks[i%4]; }
    uint16_t *d_a; uint32_t *d_m; int *d_o;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(uint16_t)));
    GPU_CHECK(hipMalloc(&d_m,N*sizeof(uint32_t)));
    GPU_CHECK(hipMalloc(&d_o,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(uint16_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_m,h_m,N*sizeof(uint32_t),hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel_test,dim3(N/BLOCK_SIZE),dim3(BLOCK_SIZE),0,0,d_a,d_m,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){
        int exp=cls_match_f16(h_a[i],h_m[i]);
        if(h_o[i]!=exp){fprintf(stderr,"FAIL[%d] val=0x%04x mask=0x%x got %d exp %d\n",
                        i,h_a[i],h_m[i],h_o[i],exp);pass=0;}
    }
    if(pass)printf("out[0..7]=%d%d%d%d%d%d%d%d\n",h_o[0],h_o[1],h_o[2],h_o[3],h_o[4],h_o[5],h_o[6],h_o[7]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_m));GPU_CHECK(hipFree(d_o));
    free(h_a);free(h_m);free(h_o); return pass?0:1;
}
