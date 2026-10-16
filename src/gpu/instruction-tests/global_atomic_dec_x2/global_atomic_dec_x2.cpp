/*
 * HIP test for GLOBAL_ATOMIC_DEC_X2 (ENC_FLAT_GLBL, 64-bit unsigned
 * decrement-wrap atomic, return-old).
 *
 *   tmp        = MEM[addr]                          (64-bit)
 *   MEM[addr]  = (tmp == 0 || tmp > DATA) ? DATA : tmp - 1
 *   VDST       = tmp                                (old value, sc0 return)
 *
 * Explicit inline asm forces the 64-bit global_atomic_dec_x2 encoding.  Each
 * thread owns a disjoint 8-byte slot (no contention).  Init values cover the
 * three branches: zero (-> data), > data (-> data), and small nonzero (-> -1).
 *
 * Build:
 *   hipcc -O2 --offload-arch=gfx950 -o global_atomic_dec_x2 \
 *         global_atomic_dec_x2.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,\
    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)

static const int N=256,BLOCK_SIZE=64;

__global__ void
kernel(uint64_t *mem, const uint64_t *data, uint64_t *old_out, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t addr = (uint64_t)(uintptr_t)(&mem[i]);
    uint64_t v = data[i], old;
    __asm__ volatile(
        "global_atomic_dec_x2 %0, %1, %2, off sc0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        : "=v"(old) : "v"(addr), "v"(v) : "memory");
    old_out[i] = old;
}

static uint64_t
ref_dec(uint64_t tmp, uint64_t d)
{
    return (tmp == 0 || tmp > d) ? d : tmp - 1;
}

int
main(void)
{
    uint64_t *h_init=(uint64_t*)malloc(N*sizeof(uint64_t));
    uint64_t *h_data=(uint64_t*)malloc(N*sizeof(uint64_t));
    uint64_t *h_mem =(uint64_t*)malloc(N*sizeof(uint64_t));
    uint64_t *h_old =(uint64_t*)malloc(N*sizeof(uint64_t));

    for (int i=0;i<N;i++){
        h_data[i] = 1000000000000ull + (uint64_t)(i%13);
        switch (i%3){
            case 0:  h_init[i] = 0ull;                       break; /* ->data */
            case 1:  h_init[i] = h_data[i] + 500ull;         break; /* >data ->data */
            default: h_init[i] = (uint64_t)((i%9)+1);        break; /* small ->-1 */
        }
    }

    uint64_t *d_mem,*d_data,*d_old;
    GPU_CHECK(hipMalloc(&d_mem, N*sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_data,N*sizeof(uint64_t)));
    GPU_CHECK(hipMalloc(&d_old, N*sizeof(uint64_t)));
    GPU_CHECK(hipMemcpy(d_mem, h_init,N*sizeof(uint64_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_data,h_data,N*sizeof(uint64_t),hipMemcpyHostToDevice));

    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_mem,d_data,d_old,N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_mem,N*sizeof(uint64_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,N*sizeof(uint64_t),hipMemcpyDeviceToHost));

    int pass=1;
    for (int i=0;i<N;i++){
        uint64_t eo=h_init[i];
        uint64_t em=ref_dec(h_init[i],h_data[i]);
        if (h_old[i]!=eo || h_mem[i]!=em){
            fprintf(stderr,"FAIL[%d] old=%llu(exp %llu) mem=%llu(exp %llu)\n",
                i,(unsigned long long)h_old[i],(unsigned long long)eo,
                (unsigned long long)h_mem[i],(unsigned long long)em);
            pass=0;break;
        }
    }
    if (pass) printf("old[0]=%llu mem[0]=%llu\n",
        (unsigned long long)h_old[0],(unsigned long long)h_mem[0]);
    printf("%s\n",pass?"PASSED":"FAILED");

    GPU_CHECK(hipFree(d_mem));GPU_CHECK(hipFree(d_data));GPU_CHECK(hipFree(d_old));
    free(h_init);free(h_data);free(h_mem);free(h_old);
    return pass?0:1;
}
