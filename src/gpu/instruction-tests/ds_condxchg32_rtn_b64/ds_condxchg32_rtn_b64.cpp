/*
 * HIP test for DS_CONDXCHG32_RTN_B64 (ENC_DS) on CDNA4 (gfx950).  Pattern C (LDS).
 *
 * Semantics (per gem5 src/arch/amdgpu/common/insts/ds.cc):
 *   ADDR0 = (ADDR + offset) & ~7   (8-byte aligned);  ADDR1 = ADDR0 + 4
 *   RETURN[0] = LDS[ADDR0];  if DATA[31]  then LDS[ADDR0] = {0, DATA[30:0]}
 *   RETURN[1] = LDS[ADDR1];  if DATA[63]  then LDS[ADDR1] = {0, DATA[62:32]}
 * i.e. two independent conditional 32-bit exchanges into an aligned 64-bit
 * LDS slot; the write happens iff the MSB of that data half is set, and the
 * stored value has its MSB cleared.  Returns the pre-exchange 64-bit value.
 *
 * Each lane owns one 8-byte LDS slot.  Low half data has MSB set (exchange
 * occurs), high half data has MSB clear (no exchange), so both the RTN value
 * and the resulting LDS contents are fully determined.
 *
 * Base encoding only (DS has no DPP/SDWA).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_condxchg32_rtn_b64 ds_condxchg32_rtn_b64.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
static const int BLOCK_SIZE=64;

__global__ void kernel(uint32_t*ret_out,uint32_t*lds_out){
    __shared__ uint32_t lds[BLOCK_SIZE*2];
    int t=threadIdx.x;
    lds[t*2+0]=0x11110000u+t;        /* old low  */
    lds[t*2+1]=0x22220000u+t;        /* old high */
    __syncthreads();

    uint32_t addr=(uint32_t)((uintptr_t)&lds[t*2]-(uintptr_t)&lds[0]);
    u32x2 data; data[0]=0x80000005u; data[1]=0x00000007u; /* lo MSB set, hi MSB clear */
    u32x2 ret;
    __asm__ volatile(
        "ds_condxchg32_rtn_b64 %0, %1, %2\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"=v"(ret):"v"(addr),"v"(data):"memory");
    __syncthreads();
    ret_out[t*2+0]=ret[0]; ret_out[t*2+1]=ret[1];
    lds_out[t*2+0]=lds[t*2+0]; lds_out[t*2+1]=lds[t*2+1];
}

int main(void){
    const int N=BLOCK_SIZE*2;
    uint32_t*h_ret=(uint32_t*)malloc(N*4),*h_lds=(uint32_t*)malloc(N*4);
    uint32_t*d_ret,*d_lds;
    GPU_CHECK(hipMalloc(&d_ret,N*4));GPU_CHECK(hipMalloc(&d_lds,N*4));
    kernel<<<1,BLOCK_SIZE>>>(d_ret,d_lds);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_ret,d_ret,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_lds,d_lds,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int t=0;t<BLOCK_SIZE&&pass;t++){
        uint32_t old0=0x11110000u+t, old1=0x22220000u+t;
        uint32_t exp_ret0=old0, exp_ret1=old1;          /* returns pre-exchange */
        uint32_t exp_lds0=0x00000005u;                  /* lo MSB set -> written, MSB cleared */
        uint32_t exp_lds1=old1;                         /* hi MSB clear -> unchanged */
        if(h_ret[t*2]!=exp_ret0||h_ret[t*2+1]!=exp_ret1||
           h_lds[t*2]!=exp_lds0||h_lds[t*2+1]!=exp_lds1){
            fprintf(stderr,"FAIL t=%d ret=%08x,%08x (exp %08x,%08x) lds=%08x,%08x (exp %08x,%08x)\n",
                t,h_ret[t*2],h_ret[t*2+1],exp_ret0,exp_ret1,
                h_lds[t*2],h_lds[t*2+1],exp_lds0,exp_lds1);
            pass=0;break;
        }
    }
    if(pass) printf("ret[0]=0x%08x lds[0]=0x%08x lds[1]=0x%08x\n",h_ret[0],h_lds[0],h_lds[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_ret));GPU_CHECK(hipFree(d_lds));free(h_ret);free(h_lds);
    return pass?0:1;
}
