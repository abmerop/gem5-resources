/* HIP test for BUFFER_ATOMIC_SMAX (32-bit VMEM atomic, return-old, gfx950).
 * Per-thread distinct 4-byte slot; host reference matches gem5 AtomicOp
 * functor. base only. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(int32_t*mem,const int32_t*data,int32_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)mem;
    u32x4 srd; srd.x=(uint32_t)(base&0xffffffffu); srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu; srd.w=0x00020000u;
    uint32_t voffset=(uint32_t)(i*4); int32_t v=data[i];
    __asm__ volatile("buffer_atomic_smax %0, %1, %2, 0 offen sc0\n\ts_waitcnt vmcnt(0)\n\t":"+v"(v):"v"(voffset),"s"(srd):"memory");
    old_out[i]=v;
}
int main(void){
    int32_t*h_i=(int32_t*)malloc(N*4),*h_d=(int32_t*)malloc(N*4),*h_m=(int32_t*)malloc(N*4),*h_o=(int32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_i[i]=(int32_t)(i*131-9000);h_d[i]=(int32_t)(-i*177+5000);}
    int32_t*d_mem,*d_data,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*4));GPU_CHECK(hipMalloc(&d_data,N*4));GPU_CHECK(hipMalloc(&d_old,N*4));
    GPU_CHECK(hipMemcpy(d_mem,h_i,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_d,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_m,d_mem,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_o,d_old,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ int32_t a=h_i[i],b=h_d[i]; int32_t em=(a>b?a:b),eo=a;
        if(h_o[i]!=eo||h_m[i]!=em){fprintf(stderr,"FAIL[%d] old=%d(e %d) mem=%d(e %d)\n",i,h_o[i],eo,h_m[i],em);pass=0;break;}}
    if(pass)printf("old[1]=%d mem[1]=%d\n",h_o[1],h_m[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
