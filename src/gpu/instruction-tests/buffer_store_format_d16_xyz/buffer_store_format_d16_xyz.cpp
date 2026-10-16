/*
 * HIP test for BUFFER_STORE_FORMAT_D16_XYZ (ENC_MUBUF, typed/format 3-component
 * packed-f16 store).  Pattern C.
 *
 * Stores three f16 components {x,y,z} (packed in v[0:1]) through a buffer
 * resource descriptor whose numeric format is FLOAT and data format is
 * 16_16_16_16, so the store is an identity pack of f16 data.  The values are
 * read back with BUFFER_LOAD_FORMAT_D16_XYZ and compared (round-trip).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o buffer_store_format_d16_xyz \
 *        buffer_store_format_d16_xyz.cpp
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
static const int N=256,BLOCK_SIZE=64;
/* stride per element = 8 bytes (4 f16 slots) so each thread owns a slot */
__global__ void kernel(uint16_t*mem,const uint16_t*xin,const uint16_t*yin,const uint16_t*zin,
                       uint16_t*xout,uint16_t*yout,uint16_t*zout,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)mem; u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu); srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    /* dst_sel xyzw=4,5,6,7 ; num_format FLOAT=7 ; data_format 16_16_16_16=11 */
    srd.w=(4u)|(5u<<3)|(6u<<6)|(7u<<9)|(7u<<12)|(11u<<15);
    uint32_t voff=(uint32_t)(i*8);
    /* pack x,y into dword0 ; z into dword1 low half */
    u32x2 data;
    data.x=((uint32_t)yin[i]<<16)|(uint32_t)xin[i];
    data.y=(uint32_t)zin[i];
    __asm__ volatile(
        "buffer_store_format_d16_xyz %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        ::"v"(data),"v"(voff),"s"(srd):"memory");
    u32x2 ld;
    __asm__ volatile(
        "buffer_load_format_d16_xyz %0, %1, %2, 0 offen\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :"=v"(ld):"v"(voff),"s"(srd):"memory");
    xout[i]=(uint16_t)(ld.x&0xffffu); yout[i]=(uint16_t)(ld.x>>16); zout[i]=(uint16_t)(ld.y&0xffffu);
}
int main(void){
    uint16_t*hx=(uint16_t*)malloc(N*2),*hy=(uint16_t*)malloc(N*2),*hz=(uint16_t*)malloc(N*2);
    uint16_t*ox=(uint16_t*)malloc(N*2),*oy=(uint16_t*)malloc(N*2),*oz=(uint16_t*)malloc(N*2);
    for(int i=0;i<N;i++){ __half hxv=__float2half((float)(i%50)); __half hyv=__float2half((float)((i%37)+1));
        __half hzv=__float2half((float)((i%23)+2));
        hx[i]=*(uint16_t*)&hxv; hy[i]=*(uint16_t*)&hyv; hz[i]=*(uint16_t*)&hzv; }
    uint16_t*dmem,*dx,*dy,*dz,*dox,*doy,*doz;
    GPU_CHECK(hipMalloc(&dmem,N*8));GPU_CHECK(hipMemset(dmem,0,N*8));
    GPU_CHECK(hipMalloc(&dx,N*2));GPU_CHECK(hipMalloc(&dy,N*2));GPU_CHECK(hipMalloc(&dz,N*2));
    GPU_CHECK(hipMalloc(&dox,N*2));GPU_CHECK(hipMalloc(&doy,N*2));GPU_CHECK(hipMalloc(&doz,N*2));
    GPU_CHECK(hipMemcpy(dx,hx,N*2,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dy,hy,N*2,hipMemcpyHostToDevice));GPU_CHECK(hipMemcpy(dz,hz,N*2,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(dmem,dx,dy,dz,dox,doy,doz,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ox,dox,N*2,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(oy,doy,N*2,hipMemcpyDeviceToHost));GPU_CHECK(hipMemcpy(oz,doz,N*2,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ if(ox[i]!=hx[i]||oy[i]!=hy[i]||oz[i]!=hz[i]){
        fprintf(stderr,"FAIL[%d] x=%u(e%u) y=%u(e%u) z=%u(e%u)\n",i,ox[i],hx[i],oy[i],hy[i],oz[i],hz[i]);pass=0;break;} }
    if(pass)printf("x[0]=%u y[0]=%u z[0]=%u\n",ox[0],oy[0],oz[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
