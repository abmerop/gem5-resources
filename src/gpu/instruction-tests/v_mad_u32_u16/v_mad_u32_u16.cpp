/* V_MAD_U32_U16 (VOP3): D.u32 = S0.u16 * S1.u16 + S2.u32 (low 16 bits of
 * S0/S1 are the multiplicands). gem5: vop3.cc. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const uint32_t*a,const uint32_t*b,const uint32_t*c,uint32_t*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 uint32_t va=a[i],vb=b[i],vc=c[i],vd;
 __asm__ volatile("v_mad_u32_u16 %0, %1, %2, %3\n\t":"=v"(vd):"v"(va),"v"(vb),"v"(vc));
 out[i]=vd; }
int main(void){
 uint32_t*ha=(uint32_t*)malloc(N*4),*hb=(uint32_t*)malloc(N*4),*hc=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*4);
 for(int i=0;i<N;i++){ha[i]=(uint32_t)(i*1009+0x70000);hb[i]=(uint32_t)((N-i)*733+0x30000);hc[i]=(uint32_t)(i*2654435761u);}
 uint32_t*da,*db,*dc,*dd;GPU_CHECK(hipMalloc(&da,N*4));GPU_CHECK(hipMalloc(&db,N*4));GPU_CHECK(hipMalloc(&dc,N*4));GPU_CHECK(hipMalloc(&dd,N*4));
 GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*4,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dc,hc,N*4,hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dc,dd,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,dd,N*4,hipMemcpyDeviceToHost));
 int pass=1;for(int i=0;i<N;i++){uint32_t e=(uint32_t)((uint16_t)ha[i])*(uint32_t)((uint16_t)hb[i])+hc[i];if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,ho[i],e);pass=0;break;}}
 if(pass)printf("out[0]=%08x out[%d]=%08x\n",ho[0],N-1,ho[N-1]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
