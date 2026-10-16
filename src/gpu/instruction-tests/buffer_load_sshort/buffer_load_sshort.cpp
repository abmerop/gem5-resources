/* HIP test for BUFFER_LOAD_SSHORT (MUBUF, signed 16-bit load, sign-extended
 * to 32 bits).  gem5 (mubuf.cc) loads a 16-bit element and sign-extends into
 * the 32-bit vdst.  Raw V# (srd) identical to the working BUFFER_LOAD_USHORT
 * test but with a 2-byte element stride; the host reference sign-extends the
 * int16 source.  Variants exercised: base */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const int16_t*src,int32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)src;u32x4 srd;srd.x=(uint32_t)(base&0xffffffffu);srd.y=(uint32_t)((base>>32)&0xffffu);srd.z=0xffffffffu;srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(4u<<12)|(4u<<15);
    uint32_t voff=(uint32_t)(i*2); int32_t d;
    __asm__ volatile("buffer_load_sshort %0, %1, %2, 0 offen\n\ts_waitcnt vmcnt(0)\n\t":"=v"(d):"v"(voff),"s"(srd):"memory");
    out[i]=d;
}
int main(void){
    int16_t*h_src=(int16_t*)malloc(N*sizeof(int16_t)); int32_t*h_out=(int32_t*)malloc(N*sizeof(int32_t));
    for(int i=0;i<N;i++)h_src[i]=(int16_t)(i*277+((i%2)?-5000:12000));
    int16_t*d_src;int32_t*d_out;GPU_CHECK(hipMalloc(&d_src,N*sizeof(int16_t)));GPU_CHECK(hipMalloc(&d_out,N*sizeof(int32_t)));
    GPU_CHECK(hipMemcpy(d_src,h_src,N*sizeof(int16_t),hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int32_t),hipMemcpyDeviceToHost));
    int pass=1;for(int i=0;i<N;i++){ int32_t e=(int32_t)h_src[i]; if(h_out[i]!=e){fprintf(stderr,"FAIL[%d] %d!=%d\n",i,h_out[i],e);pass=0;break;} }
    if(pass)printf("out[0]=%d out[1]=%d\n",h_out[0],h_out[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
