/* V_MUL_HI_U32 (ENC_VOP3). D=hi32(S0.u32*S1.u32). gem5: vop3.cc. Pattern A (vector). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
typedef uint32_t T;
#define GPU_CHECK(cmd) do{ hipError_t e=(cmd); if(e!=hipSuccess){ fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); exit(1);} }while(0)
static const int N=256,BLOCK_SIZE=64;
__global__ void kern(const T*a,const T*b,T*out,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
 T s0=a[i],s1=b[i],dv;
 __asm__ volatile("v_mul_hi_u32 %0, %1, %2\n\t":"=v"(dv):"v"(s0),"v"(s1));
 out[i]=dv; }
static T ref(T a,T b){ return ((uint32_t)(((uint64_t)a*(uint64_t)b)>>32)); }
int main(void){
 T*ha=(T*)malloc(N*sizeof(T)),*hb=(T*)malloc(N*sizeof(T)),*ho=(T*)malloc(N*sizeof(T));
 for(int i=0;i<N;i++){ha[i]=(uint32_t)(i*2654435761u);hb[i]=(uint32_t)((128-i)*40503u+123u);}
 T*da,*db,*dd;GPU_CHECK(hipMalloc(&da,N*sizeof(T)));GPU_CHECK(hipMalloc(&db,N*sizeof(T)));GPU_CHECK(hipMalloc(&dd,N*sizeof(T)));
 GPU_CHECK(hipMemcpy(da,ha,N*sizeof(T),hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(db,hb,N*sizeof(T),hipMemcpyHostToDevice));
 kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(da,db,dd,N);GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
 GPU_CHECK(hipMemcpy(ho,dd,N*sizeof(T),hipMemcpyDeviceToHost));
 int pass=1;for(int i=0;i<N;i++){T e=ref(ha[i],hb[i]);if(ho[i]!=e){fprintf(stderr,"FAIL[%d] got %08x exp %08x\n",i,ho[i],e);pass=0;break;}}
 if(pass)printf("out[1]=%08x\n",ho[1]);printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1; }
