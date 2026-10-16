#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do { hipError_t e=(cmd); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);}} while(0)

static const int N=256, BLK=64;
__global__ void k(const float*a,const float*b,float*ob,float*od,float*os,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float va=a[i], vb=b[i], r;
    __asm__ volatile("v_add_f32_e32 %0, %1, %2":"=v"(r):"v"(va),"v"(vb)); ob[i]=r;
    float rd; __asm__ volatile("v_add_f32_dpp %0,%1,%2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf":"=v"(rd):"v"(va),"v"(vb)); od[i]=rd;
    float rs; __asm__ volatile("v_add_f32_sdwa %0,%1,%2 dst_sel:DWORD src0_sel:DWORD src1_sel:DWORD":"=v"(rs):"v"(va),"v"(vb)); os[i]=rs;
}
int main(void){
    float*ha=(float*)malloc(N*4),*hb=(float*)malloc(N*4);
    float*hob=(float*)malloc(N*4),*hod=(float*)malloc(N*4),*hos=(float*)malloc(N*4);
    for(int i=0;i<N;i++){ ha[i]=(float)(i%64); hb[i]=100.0f; }
    float*da,*db,*dob,*dod,*dos;
    GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));
    GPU_CHECK(hipMalloc(&dob,N*4));GPU_CHECK(hipMalloc(&dod,N*4));GPU_CHECK(hipMalloc(&dos,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k,dim3(N/BLK),dim3(BLK),0,0,da,db,dob,dod,dos,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(hob,dob,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hod,dod,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hos,dos,N*4,hipMemcpyDeviceToHost));
    const int perm[4]={1,0,3,2}; int pass=1;
    for(int i=0;i<N&&pass;i++){
        float eb=ha[i]+hb[i];
        int lane=i%64, base=i-(lane&3), sl=base+perm[lane&3];
        float ed=ha[sl]+hb[i];
        float es=ha[i]+hb[i];
        if(hob[i]!=eb){fprintf(stderr,"FAIL base[%d] %.1f!=%.1f\n",i,hob[i],eb);pass=0;break;}
        if(hod[i]!=ed){fprintf(stderr,"FAIL dpp[%d] %.1f!=%.1f\n",i,hod[i],ed);pass=0;break;}
        if(hos[i]!=es){fprintf(stderr,"FAIL sdwa[%d] %.1f!=%.1f\n",i,hos[i],es);pass=0;break;}
    }
    printf("v_add_f32 base+dpp+sdwa\n"); printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
