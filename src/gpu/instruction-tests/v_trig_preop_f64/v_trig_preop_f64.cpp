/*
 * HIP test for V_TRIG_PREOP_F64 (VOP3).
 *
 * D.f64 = ldexp( 53-bit segment of 2/PI selected by S1[4:0], scaled by the
 * exponent of S0 ). Used for trigonometric argument reduction; rounding is
 * toward zero. See gem5 src/arch/amdgpu/common/insts/vop3.cc.
 *
 * The host reference reproduces the exact integer lookup (same 1201-bit 2/PI
 * table gem5 uses). For normal inputs the result is bit-exact, but real
 * hardware stores a shorter 2/PI ROM and drops low-order bits for very large
 * exponents (its low bits fall off the end of the ROM), so large-exponent
 * cases are accepted within a small relative tolerance rather than
 * bit-for-bit. gem5 models the full-precision segment and matches the
 * reference exactly.
 */
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,        \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void kern(const double *a, const int *sel, double *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double s=a[i]; int seg=sel[i]; double d;
    __asm__ volatile("v_trig_preop_f64 %0, %1, %2\n\t"
                     :"=v"(d):"v"(s),"v"(seg));
    out[i]=d;
}

/*
 * First 1201 fractional bits of 2/PI, little-endian uint32 words (word w holds
 * fractional bits [32*w : 32*w+31]; b_1200 is the MSB of the fraction). This
 * is floor((2/PI) * 2^1201) and matches the gem5 table.
 */
static const uint32_t two_over_pi_w[] = {
    0x608df8f6U, 0xba10ac06U, 0xbf623f1aU, 0x25d4d7f6U, 0x73ef14a5U, 0xe2f67a0eU,
    0x758fd7cbU, 0xd45aea4fU, 0x7ecd3cbfU, 0x136e9e8cU, 0xfd9e4f96U, 0xda3eda6cU,
    0x2316b414U, 0x301fde5eU, 0x2fffbc0bU, 0x50763ff1U, 0xbf177bf2U, 0x73e93908U,
    0xac7306a6U, 0xfc827323U, 0xe04d68beU, 0x8909d338U, 0x6bea5d76U, 0x4e7dd104U,
    0xd6396253U, 0x2439fc3bU, 0x5dd413a3U, 0xa5c00c92U, 0xdc74849bU, 0x8ac36e48U,
    0xc757bd77U, 0x2083fca2U, 0x2b327887U, 0xbb81b6c5U, 0xafa3ea69U, 0x2a53f84eU,
    0x06dc9c88U, 0x000145f3U,
};

static int getbit(int idx){
    if(idx<0||idx>1200) return 0;
    return (two_over_pi_w[idx>>5]>>(idx&31))&1;
}

static double host_ref(double x, int s1){
    uint64_t u; memcpy(&u,&x,sizeof(u));
    int exponent = (int)((u>>52)&0x7ff);
    int shift = (s1&0x1f)*53;
    if(exponent>1077) shift += exponent-1077;
    uint64_t r=0;
    for(int i=0;i<53;i++) r |= (uint64_t)getbit(1148-shift+i)<<i;
    int scale=-53-shift;
    if(exponent>=1968) scale+=128;
    return ldexp((double)r, scale);
}

/* Accept a bit-exact match (normal inputs) or, for large-exponent inputs
 * where hardware's shorter 2/PI ROM drops low-order bits, a small relative
 * tolerance. The worst case in this test (exp 1000, seg 4) loses ~28 low
 * bits: a ~3e-8 relative error. */
static int close_tp(double g,double e){
    uint64_t bg,be; memcpy(&bg,&g,8); memcpy(&be,&e,8);
    if(bg==be) return 1;
    if(e==0.0) return g==0.0;
    return fabs(g-e) <= 1e-6*fabs(e);
}

int main(void){
    size_t szd=N*sizeof(double), szi=N*sizeof(int);
    double *h_a=(double*)malloc(szd),*h_o=(double*)malloc(szd);
    int    *h_s=(int*)malloc(szi);
    // Cover a range of exponents (including >1077 and >=1968, which trigger
    // the extra scaling paths) and several segment selects.
    int expos[]={-40,-4,0,4,60,300,900,1000};
    int nexp=(int)(sizeof(expos)/sizeof(expos[0]));
    for(int i=0;i<N;i++){
        int e=expos[i%nexp];
        h_a[i]=ldexp(1.0+0.5*((i/nexp)%2), e);   // 1.0 or 1.5 mantissa
        h_s[i]=(i/nexp)%5;                        // segment selects 0..4
    }
    double *d_a,*d_o; int *d_s;
    GPU_CHECK(hipMalloc(&d_a,szd));GPU_CHECK(hipMalloc(&d_o,szd));
    GPU_CHECK(hipMalloc(&d_s,szi));
    GPU_CHECK(hipMemcpy(d_a,h_a,szd,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_s,h_s,szi,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_s,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,szd,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ double e=host_ref(h_a[i],h_s[i]);
        if(!close_tp(h_o[i],e)){
            fprintf(stderr,"FAIL[%d] in=%a s1=%d got %a exp %a\n",
                    i,h_a[i],h_s[i],h_o[i],e);pass=0;break;} }
    if(pass)printf("out[0]=%a out[%d]=%a\n",h_o[0],N-1,h_o[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_o));GPU_CHECK(hipFree(d_s));
    free(h_a);free(h_o);free(h_s); return pass?0:1;
}
