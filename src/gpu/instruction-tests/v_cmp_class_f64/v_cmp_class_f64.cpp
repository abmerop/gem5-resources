/*
 * HIP test for V_CMP_CLASS_F64 (v_cmp_class_f64_e64), CDNA4 / gfx950.
 * V_CMP_CLASS: result = (float-class(S0) intersects the class bitmask S1) ? 1:0.
 * Host reference mirrors gem5's exact 10-bit class test. A spread of values
 * (+/-normal, +/-zero, +/-inf, NaN, finite) and per-lane masks are used.
 * Variants exercised: base
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){\
 fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256, BLOCK_SIZE=64;

static int cls_match(double x, uint32_t m){
    if((m&3u) && isnan(x)) return 1;
    if((m&(1u<<2)) && isinf(x)&&signbit(x)) return 1;
    if((m&(1u<<3)) && isnormal(x)&&signbit(x)) return 1;
    if((m&(1u<<4)) && fpclassify(x)==FP_SUBNORMAL&&signbit(x)) return 1;
    if((m&(1u<<5)) && fpclassify(x)==FP_ZERO&&signbit(x)) return 1;
    if((m&(1u<<6)) && fpclassify(x)==FP_ZERO&&!signbit(x)) return 1;
    if((m&(1u<<7)) && fpclassify(x)==FP_SUBNORMAL&&!signbit(x)) return 1;
    if((m&(1u<<8)) && isnormal(x)&&!signbit(x)) return 1;
    if((m&(1u<<9)) && isinf(x)&&!signbit(x)) return 1;
    return 0;
}

__global__ void
kernel_test(const double *a, const uint32_t *mask_in, int *out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    double x=a[i]; uint32_t m=mask_in[i];
    uint64_t mask;
    __asm__ volatile("v_cmp_class_f64_e64 %0, %1, %2\n\t":"=s"(mask):"v"(x),"v"(m));
    out[i]=(int)((mask>>threadIdx.x)&1ULL);
}

int main(void){
    double *h_a=(double*)malloc(N*sizeof(double));
    uint32_t *h_m=(uint32_t*)malloc(N*sizeof(uint32_t));
    int *h_o=(int*)malloc(N*sizeof(int));
    double vals[8];
    vals[0]=(double)3.5;        /* +normal */
    vals[1]=(double)-3.5;       /* -normal */
    vals[2]=(double)0.0;        /* +zero   */
    vals[3]=(double)-0.0;       /* -zero   */
    vals[4]=(double)INFINITY;   /* +inf    */
    vals[5]=(double)-INFINITY;  /* -inf    */
    vals[6]=(double)NAN;        /* NaN     */
    vals[7]=(double)42.0;       /* +normal */
    uint32_t masks[4]={0x1FF,0x040|0x020,0x200|0x004,0x100|0x008};
    for(int i=0;i<N;i++){ h_a[i]=vals[i%8]; h_m[i]=masks[i%4]; }
    double *d_a; uint32_t *d_m; int *d_o;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(double)));GPU_CHECK(hipMalloc(&d_m,N*sizeof(uint32_t)));GPU_CHECK(hipMalloc(&d_o,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(double),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_m,h_m,N*sizeof(uint32_t),hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel_test,dim3(N/BLOCK_SIZE),dim3(BLOCK_SIZE),0,0,d_a,d_m,d_o,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N&&pass;i++){
        int exp=cls_match(h_a[i],h_m[i]);
        if(h_o[i]!=exp){fprintf(stderr,"FAIL[%d] val#%d mask=0x%x got %d exp %d\n",i,i%8,h_m[i],h_o[i],exp);pass=0;}
    }
    if(pass)printf("out[0..7]=%d%d%d%d%d%d%d%d\n",h_o[0],h_o[1],h_o[2],h_o[3],h_o[4],h_o[5],h_o[6],h_o[7]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_m));GPU_CHECK(hipFree(d_o));
    free(h_a);free(h_m);free(h_o); return pass?0:1;
}
