#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int BLOCK_SIZE=64;

/* S_STORE_DWORD (ENC_SMEM): MEM[base+off] = SDATA. Store, verify readback.
 * gem5: scalar store hits GPU_PA Ruby protocol assert (GL0K_Wakeup). */
__global__ void kernel(uint32_t*mem,uint32_t v0){
    uint64_t base=(uint64_t)mem; uint32_t d=v0;
    /* glc sets the SMEM GLC bit -> GroupCoherent -> DLC on the request, which
     * resolves the GL2 policy to CACHE_BYPASS so the scalar store propagates to
     * memory (rather than staying dirty in GL2, invisible to the D2H readback). */
    __asm__ volatile("s_store_dword %0, %1, 0 glc\n\t"
                     ::"s"(d),"s"(base):"memory");
}
int main(void){
    uint32_t h=0, v0=0xABCD1234u; uint32_t*d_mem;
    GPU_CHECK(hipMalloc(&d_mem,4)); GPU_CHECK(hipMemcpy(d_mem,&h,4,hipMemcpyHostToDevice));
    kernel<<<1,BLOCK_SIZE>>>(d_mem,v0);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(&h,d_mem,4,hipMemcpyDeviceToHost));
    int pass=(h==v0);
    if(!pass)fprintf(stderr,"FAIL got 0x%08x exp 0x%08x\n",h,v0);
    else printf("mem=0x%08x\n",h);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
