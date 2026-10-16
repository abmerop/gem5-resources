/*
 * HIP test for V_CVT_PKNORM_I16_F16 (ENC_VOP3, pack two normalized f16 -> two
 * signed i16), pattern A, base only.
 *
 * Architectural semantics: each f16 source (src0 low16, src1 low16) is in
 * [-1.0, 1.0]; the result is round(clamp(v,-1,1) * 32767) as a signed int16,
 * packed as (norm(src1) << 16) | (norm(src0) & 0xffff).
 *
 * Inputs are f16 values that map to whole int16 results so the check is exact
 * regardless of the rounding mode (values chosen so v*32767 is an integer or
 * effectively saturates cleanly).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const _Float16*a,const _Float16*b,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    _Float16 s0=a[i],s1=b[i]; uint32_t d;
    __asm__ volatile("v_cvt_pknorm_i16_f16 %0, %1, %2\n\t":"=v"(d):"v"(s0),"v"(s1));
    out[i]=d;
}

static int16_t norm(float v){ if(v>1.0f)v=1.0f; if(v<-1.0f)v=-1.0f; return (int16_t)lrintf(v*32767.0f); }

int main(void){
    /* f16 values exactly representable, chosen so v*32767 rounds cleanly */
    static const float pool[]={0.0f,1.0f,-1.0f,0.5f,-0.5f,0.25f,-0.25f,0.75f,
                               -0.75f,0.125f,-0.125f,0.375f,-0.375f,1.0f,-1.0f,0.0f};
    const int P=(int)(sizeof(pool)/sizeof(pool[0]));
    _Float16*h_a=(_Float16*)malloc(N*2),*h_b=(_Float16*)malloc(N*2);
    uint32_t*h_out=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_a[i]=(_Float16)pool[i%P];h_b[i]=(_Float16)pool[(i*5+2)%P];}
    _Float16*d_a,*d_b;uint32_t*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*2));GPU_CHECK(hipMalloc(&d_b,N*2));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*2,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*2,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t e=(((uint32_t)(uint16_t)norm((float)h_b[i]))<<16)|((uint32_t)(uint16_t)norm((float)h_a[i]));
        if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] got 0x%08x exp 0x%08x (a=%g b=%g)\n",i,h_out[i],e,(float)h_a[i],(float)h_b[i]);pass=0;break;}}
    if(pass)printf("out[0]=0x%08x out[1]=0x%08x\n",h_out[0],h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
