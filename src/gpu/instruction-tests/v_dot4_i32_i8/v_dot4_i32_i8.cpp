#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);}} while(0)

static const int N=256, BLK=64;
__global__ void k(const uint32_t*a,const uint32_t*b,const int*c,int*o,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint32_t va=a[i],vb=b[i]; int vc=c[i],r;
    __asm__ volatile("v_dot4_i32_i8 %0, %1, %2, %3":"=v"(r):"v"(va),"v"(vb),"v"(vc)); o[i]=r;
}
int main(void){
    uint32_t*ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4); int*hc=(int*)malloc(N*4),*ho=(int*)malloc(N*4);
    for(int i=0;i<N;i++){
        /* Scaled so |product| > 127, exercising full-width accumulation. */
        int8_t a0=(i%5-2)*40,a1=(i%7-3)*30,a2=(i%3-1)*60,a3=(i%4-2)*40;
        int8_t b0=(i%6-3)*30,b1=(i%5-2)*40,b2=(i%3)*50,b3=(i%4-1)*40;
        ha[i]=(uint8_t)a0|((uint8_t)a1<<8)|((uint8_t)a2<<16)|((uint8_t)a3<<24);
        hb[i]=(uint8_t)b0|((uint8_t)b1<<8)|((uint8_t)b2<<16)|((uint8_t)b3<<24);
        hc[i]=i%11;
    }
    uint32_t*da,*db; int*dc,*dbuf; GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&dbuf,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k,dim3(N/BLK),dim3(BLK),0,0,da,db,dc,dbuf,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dbuf,N*4,hipMemcpyDeviceToHost));
    int pass=1; for(int i=0;i<N&&pass;i++){
        int e=hc[i];
        for(int k=0;k<4;k++){ int8_t av=(int8_t)((ha[i]>>(8*k))&0xff), bv=(int8_t)((hb[i]>>(8*k))&0xff); e+=(int)av*(int)bv; }
        if(ho[i]!=e){fprintf(stderr,"FAIL[%d] %d!=%d\n",i,ho[i],e);pass=0;}
    }
    printf("v_dot4_i32_i8 sum4+c\n"); printf("%s\n",pass?"PASSED":"FAILED"); return pass?0:1;
}
