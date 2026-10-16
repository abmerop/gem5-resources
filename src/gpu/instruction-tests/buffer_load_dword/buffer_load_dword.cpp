/*
 * HIP test for BUFFER_LOAD_DWORD (ENC_MUBUF, load 32-bit through a buffer
 * resource descriptor with per-lane voffset).  Verified against src[i].
 * Pattern C, base only.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(const uint32_t*src,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)src;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(4u<<12)|(4u<<15);
    uint32_t voffset=(uint32_t)(i*4);
    uint32_t data;
    __asm__ volatile(
        "buffer_load_dword %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :"=v"(data):"v"(voffset),"s"(srd):"memory");
    out[i]=data;
}
int main(void){
    uint32_t*h=(uint32_t*)malloc(N*4),*o=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++) h[i]=(uint32_t)(i*40503u+123456u);
    uint32_t*d,*dO; GPU_CHECK(hipMalloc(&d,N*4));GPU_CHECK(hipMalloc(&dO,N*4));
    GPU_CHECK(hipMemcpy(d,h,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(o,dO,N*4,hipMemcpyDeviceToHost));
    int pass=1; for(int i=0;i<N;i++) if(o[i]!=h[i]){fprintf(stderr,"FAIL[%d] 0x%08x exp 0x%08x\n",i,o[i],h[i]);pass=0;break;}
    if(pass)printf("out[0]=0x%08x out[255]=0x%08x\n",o[0],o[255]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
