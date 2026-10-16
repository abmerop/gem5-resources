/*
 * HIP test for S_BUFFER_STORE_DWORD (ENC_SMEM, scalar buffer store).
 * Pattern B, base only.
 *
 * S_BUFFER_STORE_DWORD writes one SGPR dword through a buffer resource
 * descriptor (SBASE[0:3]) at a scalar byte offset (SOFFSET).  This test builds
 * a linear buffer descriptor over the output array (stride 0, large
 * num_records) and has each wavefront store a known scalar value at its own
 * dword offset; the host reads the array back and verifies every slot.  The
 * compiler does not emit S_BUFFER_STORE_DWORD for HIP global stores, so it is
 * placed via inline asm and followed by s_waitcnt lgkmcnt(0).
 *
 * Assembler form (validated with llvm-mc -mcpu=gfx950):
 *   s_buffer_store_dword s5, s[0:3], 0x0
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o s_buffer_store_dword \
 *         s_buffer_store_dword.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d '%s'\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int BLOCK_SIZE=64;      /* one wavefront per block */
static const int NWF=64;             /* number of wavefronts / dwords */

__global__ void kernel(uint32_t*out,int nwf){
    int wf=blockIdx.x; if(wf>=nwf)return;
    uint64_t base=(uint64_t)out;
    u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu);
    srd.y=(uint32_t)((base>>32)&0xffffu);   /* stride=0 in [61:48] */
    srd.z=0xffffffffu;                        /* num_records (bytes) */
    srd.w=0x00020000u;                        /* default data format flags */
    uint32_t val=(uint32_t)(wf*2246822519u+13u);
    uint32_t off=(uint32_t)(wf*4);
    __asm__ volatile("s_buffer_store_dword %0, %1, %2\n\t"
                     "s_waitcnt lgkmcnt(0)\n\t"
                     ::"s"(val),"s"(srd),"s"(off):"memory");
}
int main(void){
    uint32_t*h_out=(uint32_t*)malloc(NWF*4);
    for(int i=0;i<NWF;i++) h_out[i]=0xdeadbeefu;
    uint32_t*d_out;
    GPU_CHECK(hipMalloc(&d_out,NWF*4));
    GPU_CHECK(hipMemcpy(d_out,h_out,NWF*4,hipMemcpyHostToDevice));
    kernel<<<NWF,BLOCK_SIZE>>>(d_out,NWF);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,NWF*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<NWF;i++){uint32_t exp=(uint32_t)(i*2246822519u+13u);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got 0x%x exp 0x%x\n",
            i,h_out[i],exp);pass=0;break;}}
    if(pass)printf("out[0]=0x%x out[%d]=0x%x\n",h_out[0],NWF-1,h_out[NWF-1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
