/*
 * HIP test for BUFFER_ATOMIC_ADD (ENC_MUBUF, 32-bit integer add atomic,
 * return-old).  Pattern C.
 *
 *   tmp        = MEM[addr]
 *   MEM[addr]  = tmp + DATA            (32-bit integer add)
 *   VDATA      = tmp                   (old value, with sc0 return)
 *
 * MUBUF addresses memory through a 128-bit buffer resource descriptor (V#)
 * in 4 consecutive SGPRs.  Each thread targets its own 4-byte slot (offen)
 * so old/new values are deterministic.  vdata is "+v" (addend in, old out).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o buffer_atomic_add buffer_atomic_add.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(uint32_t*mem,const uint32_t*data,uint32_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)mem;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    srd.w=0x00020000u;
    uint32_t voffset=(uint32_t)(i*4);
    uint32_t v=data[i];
    __asm__ volatile(
        "buffer_atomic_add %0, %1, %2, 0 offen sc0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :"+v"(v):"v"(voffset),"s"(srd):"memory");
    old_out[i]=v;
}
int main(void){
    uint32_t*h_init=(uint32_t*)malloc(N*4),*h_data=(uint32_t*)malloc(N*4);
    uint32_t*h_mem=(uint32_t*)malloc(N*4),*h_old=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){h_init[i]=(uint32_t)(i*1000+1);h_data[i]=(uint32_t)(i*7+3);}
    uint32_t*d_mem,*d_data,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*4));GPU_CHECK(hipMalloc(&d_data,N*4));GPU_CHECK(hipMalloc(&d_old,N*4));
    GPU_CHECK(hipMemcpy(d_mem,h_init,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_data,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_mem,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t exp_old=h_init[i],exp_mem=h_init[i]+h_data[i];
        if(h_old[i]!=exp_old||h_mem[i]!=exp_mem){
            fprintf(stderr,"FAIL[%d] old=%u(exp %u) mem=%u(exp %u)\n",i,h_old[i],exp_old,h_mem[i],exp_mem);pass=0;break;}
    }
    if(pass)printf("old[1]=%u mem[1]=%u\n",h_old[1],h_mem[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
