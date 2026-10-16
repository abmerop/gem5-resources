/*
 * HIP test for BUFFER_ATOMIC_ADD_F32 (ENC_MUBUF, 32-bit float add atomic,
 * return-old).  Pattern C.
 *
 *   tmp        = MEM[addr]
 *   MEM[addr]  = tmp + DATA            (32-bit float add)
 *   VDATA      = tmp                   (old value, with sc0 return)
 *
 * MUBUF addresses memory through a 128-bit buffer resource descriptor (V#)
 * in 4 consecutive SGPRs.  Each thread targets its own 4-byte slot (offen)
 * so there is no contention and old/new values are deterministic.  vdata is
 * "+v" (addend in, old out).  All test values are small integers exactly
 * representable in float, so the float add is exact.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o buffer_atomic_add_f32 buffer_atomic_add_f32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int N=256,BLOCK_SIZE=64;
__global__ void kernel(float*mem,const float*data,float*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t base=(uint64_t)mem;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu;
    srd.w=0x00020000u;
    uint32_t voffset=(uint32_t)(i*4);
    float v=data[i];
    __asm__ volatile(
        "buffer_atomic_add_f32 %0, %1, %2, 0 offen sc0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        :"+v"(v):"v"(voffset),"s"(srd):"memory");
    old_out[i]=v;
}
int main(void){
    float*h_init=(float*)malloc(N*4),*h_data=(float*)malloc(N*4);
    float*h_mem=(float*)malloc(N*4),*h_old=(float*)malloc(N*4);
    for(int i=0;i<N;i++){h_init[i]=(float)(i*3+1);h_data[i]=(float)(i*7+2);}
    float*d_mem,*d_data,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*4));GPU_CHECK(hipMalloc(&d_data,N*4));GPU_CHECK(hipMalloc(&d_old,N*4));
    GPU_CHECK(hipMemcpy(d_mem,h_init,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_data,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_mem,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        float exp_old=h_init[i],exp_mem=h_init[i]+h_data[i];
        if(fabsf(h_old[i]-exp_old)>1e-3f||fabsf(h_mem[i]-exp_mem)>1e-3f){
            fprintf(stderr,"FAIL[%d] old=%f(exp %f) mem=%f(exp %f)\n",i,h_old[i],exp_old,h_mem[i],exp_mem);pass=0;break;}
    }
    if(pass)printf("old[1]=%f mem[1]=%f\n",h_old[1],h_mem[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
