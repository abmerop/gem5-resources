/* HIP test for FLAT_ATOMIC_CMPSWAP (32-bit VMEM compare-and-swap, gfx950).
 *
 * gem5 (flat.cc -> AtomicOpCAS): tmp=MEM; if(MEM==cmp) MEM=src; RETURN=tmp.
 * The data operand is a 32-bit pair {src (low dword), cmp (high dword)} (same
 * layout as the working flat_atomic_cmpswap_x2 test).  Per-lane distinct slot;
 * half the lanes have cmp==mem (swap happens), half don't (unchanged).
 * Variants exercised: base */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(uint32_t*mem,const uint32_t*src,const uint32_t*cmp,
                       uint32_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(&mem[i]);
    u32x2 data; data.x=src[i]; data.y=cmp[i]; uint32_t old;
    __asm__ volatile("flat_atomic_cmpswap %0, %1, %2 sc0\n\ts_waitcnt vmcnt(0)\n\t"
                     :"=v"(old):"v"(addr),"v"(data):"memory");
    old_out[i]=old;
}
int main(void){
    uint32_t*h_i=(uint32_t*)malloc(N*4),*h_s=(uint32_t*)malloc(N*4),
             *h_c=(uint32_t*)malloc(N*4),*h_m=(uint32_t*)malloc(N*4),*h_o=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){ h_i[i]=0x5000u+i*13u; h_s[i]=0xAA000000u+i;
        h_c[i]=(i&1)?h_i[i]:(h_i[i]^0x2u); }
    uint32_t*d_mem,*d_src,*d_cmp,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*4));GPU_CHECK(hipMalloc(&d_src,N*4));
    GPU_CHECK(hipMalloc(&d_cmp,N*4));GPU_CHECK(hipMalloc(&d_old,N*4));
    GPU_CHECK(hipMemcpy(d_mem,h_i,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_src,h_s,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_cmp,h_c,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_src,d_cmp,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_m,d_mem,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_o,d_old,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint32_t eo=h_i[i], em=(h_i[i]==h_c[i])?h_s[i]:h_i[i];
        if(h_o[i]!=eo||h_m[i]!=em){fprintf(stderr,"FAIL[%d] old=%08x(e %08x) mem=%08x(e %08x)\n",
            i,h_o[i],eo,h_m[i],em);pass=0;break;} }
    if(pass)printf("old[0]=%08x mem[0]=%08x mem[1]=%08x\n",h_o[0],h_m[0],h_m[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
