/*
 * HIP test for GLOBAL_ATOMIC_CMPSWAP_X2 (ENC_FLAT_GLBL, 64-bit compare-and-swap,
 * return-old).  Pattern C, base only.
 *
 *   tmp = MEM64[addr];
 *   MEM64[addr] = (tmp == cmp) ? swap : tmp;
 *   VDST64 = tmp                              (old value, sc0 return)
 *
 * The data operand is 4 consecutive VGPRs: { swap:u64 (v[0:1]),
 * cmp:u64 (v[2:3]) } -- same ordering as the 32-bit GLOBAL_ATOMIC_CMPSWAP
 * ({swap, cmp}) widened to 64-bit.  64-bit MUBUF/FLAT CAS works in gem5
 * (cf. BUFFER_ATOMIC_CMPSWAP_X2); the old "64-bit _x2 CAS buggy" note is
 * stale.
 *
 * Each thread targets a disjoint 8-byte slot (no contention).  Even lanes use
 * cmp == init so the swap succeeds (mem becomes swap); odd lanes use
 * cmp != init so the swap fails (mem stays init).  Returned old == init always.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint64_t u64x2 __attribute__((ext_vector_type(2)));
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(uint64_t*mem,const uint64_t*swapv,const uint64_t*cmpv,
                       uint64_t*old_out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint64_t addr=(uint64_t)(uintptr_t)(&mem[i]);
    u64x2 data; data.x=swapv[i]; data.y=cmpv[i];   /* {swap, cmp} */
    uint64_t old;
    __asm__ volatile("global_atomic_cmpswap_x2 %0, %1, %2, off sc0\n\t"
                     "s_waitcnt vmcnt(0)\n\t"
                     :"=v"(old):"v"(addr),"v"(data):"memory");
    old_out[i]=old;
}
int main(void){
    uint64_t*h_i=(uint64_t*)malloc(N*8),*h_s=(uint64_t*)malloc(N*8),*h_c=(uint64_t*)malloc(N*8);
    uint64_t*h_old=(uint64_t*)malloc(N*8),*h_m=(uint64_t*)malloc(N*8);
    for(int i=0;i<N;i++){
        h_i[i]=(uint64_t)(i)*0x100000001ull+7ull;
        h_s[i]=(uint64_t)(i)*0x00A0000B0ull+0x50000ull;
        h_c[i]=(i&1)?(h_i[i]^0xdeadbeefdeadbeefull):h_i[i];  /* even: swap fires */
    }
    uint64_t*d_mem,*d_s,*d_c,*d_old;
    GPU_CHECK(hipMalloc(&d_mem,N*8));GPU_CHECK(hipMalloc(&d_s,N*8));GPU_CHECK(hipMalloc(&d_c,N*8));GPU_CHECK(hipMalloc(&d_old,N*8));
    GPU_CHECK(hipMemcpy(d_mem,h_i,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_s,h_s,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*8,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_s,d_c,d_old,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_old,d_old,N*8,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_m,d_mem,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){ uint64_t em=(i&1)?h_i[i]:h_s[i];
        if(h_old[i]!=h_i[i]||h_m[i]!=em){
            fprintf(stderr,"FAIL[%d] old=%llu(e%llu) mem=%llu(e%llu)\n",i,
                (unsigned long long)h_old[i],(unsigned long long)h_i[i],
                (unsigned long long)h_m[i],(unsigned long long)em);pass=0;break;} }
    if(pass)printf("old[0]=%llu mem[0]=%llu mem[1]=%llu\n",
        (unsigned long long)h_old[0],(unsigned long long)h_m[0],(unsigned long long)h_m[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
