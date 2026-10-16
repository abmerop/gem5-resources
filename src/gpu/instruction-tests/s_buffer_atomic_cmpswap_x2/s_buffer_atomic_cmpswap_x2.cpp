/*
 * HIP test for S_BUFFER_ATOMIC_CMPSWAP_X2 (SMEM 64-bit scalar-buffer
 * compare-and-swap, return-old).  Pattern B, base only.
 *
 * gem5 (vega/insts/smem_atomic.hh, Inst_SMEM_BUFFER_ATOMIC_CMPSWAP<u64>):
 * SDATA is a 4-dword pair { cmp:u64 (SDATA[0:1]), new/swap:u64 (SDATA[2:3]) }.
 *   MEM64 = (MEM64 == cmp) ? swap : MEM64;
 * with GLC set (AtomicReturn) the old MEM64 is written back to SDATA[0:1].
 *
 * 64-bit scalar-buffer atomics work in gem5 (unlike the broken 32-bit path;
 * cf. S_BUFFER_ATOMIC_DEC_X2).  Single wave, one 64-bit slot addressed via a
 * raw buffer resource descriptor (V#).  Seed mem == cmp so the swap fires:
 *   init=cmp=0x00000007_0000002A, swap=0x00000011_00000063
 *   -> old = init, mem = swap.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

__global__ void kernel(uint64_t*mem,uint64_t cmp,uint64_t swap,uint64_t*old_out){
    uint64_t base=(uint64_t)mem; u32x4 srd;
    srd.x=(uint32_t)(base&0xffffffffu); srd.y=(uint32_t)((base>>32)&0xffffu);
    srd.z=0xffffffffu; srd.w=0x00020000u;
    /* sdata = { cmp:u64 [0:1], swap:u64 [2:3] }; old returned in [0:1] */
    u32x4 sdata;
    sdata.x=(uint32_t)(cmp&0xffffffffu);  sdata.y=(uint32_t)(cmp>>32);
    sdata.z=(uint32_t)(swap&0xffffffffu); sdata.w=(uint32_t)(swap>>32);
    __asm__ volatile("s_buffer_atomic_cmpswap_x2 %0, %1, 0 glc\n\t"
                     "s_waitcnt lgkmcnt(0)\n\t"
                     :"+s"(sdata):"s"(srd):"memory");
    if(threadIdx.x==0) old_out[0]=((uint64_t)sdata.y<<32)|sdata.x;
}
int main(void){
    uint64_t init=0x000000070000002Aull, cmp=init, swap=0x0000001100000063ull;
    uint64_t *d_mem,*d_old, h_mem, h_old;
    GPU_CHECK(hipMalloc(&d_mem,8));GPU_CHECK(hipMalloc(&d_old,8));
    GPU_CHECK(hipMemcpy(d_mem,&init,8,hipMemcpyHostToDevice));
    kernel<<<1,64>>>(d_mem,cmp,swap,d_old);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(&h_old,d_old,8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(&h_mem,d_mem,8,hipMemcpyDeviceToHost));
    uint64_t exp_old=init, exp_mem=swap;
    int pass=(h_old==exp_old && h_mem==exp_mem);
    if(!pass) fprintf(stderr,"FAIL old=%llu(e%llu) mem=%llu(e%llu)\n",
        (unsigned long long)h_old,(unsigned long long)exp_old,
        (unsigned long long)h_mem,(unsigned long long)exp_mem);
    else printf("old=%llu mem=%llu\n",(unsigned long long)h_old,(unsigned long long)h_mem);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
