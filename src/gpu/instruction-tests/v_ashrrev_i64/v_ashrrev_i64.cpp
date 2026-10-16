/*
 * HIP test for V_ASHRREV_I64 (ENC_VOP3, reverse arithmetic shift right, 64-bit).
 * gem5 semantics: D.i64 = S1.i64 >> (S0.u & 63)  (sign-replicating shift).
 * "rev" -> the shift amount is the first source (S0), the value is S1.
 * Inline asm forces the VOP3 encoding; value in a 64-bit VGPR pair.
 * Exact integer compare.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const int64_t *val, const uint32_t *sh, int64_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t s1=val[i], d; uint32_t s0=sh[i];
    __asm__ volatile("v_ashrrev_i64 %0, %1, %2\n\t"
                     : "=v"(d) : "v"(s0), "v"(s1));
    out[i] = d;
}

int main(void)
{
    int64_t *h_val=(int64_t*)malloc(N*8),*h_out=(int64_t*)malloc(N*8);
    uint32_t *h_sh=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){
        h_val[i]=(int64_t)(i*1103515245LL+12345) ^ ((int64_t)(i&1?-1:1)<<50);
        h_sh[i]=(uint32_t)(i%64);
    }
    int64_t *d_val,*d_out; uint32_t *d_sh;
    GPU_CHECK(hipMalloc(&d_val,N*8));GPU_CHECK(hipMalloc(&d_sh,N*4));GPU_CHECK(hipMalloc(&d_out,N*8));
    GPU_CHECK(hipMemcpy(d_val,h_val,N*8,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_sh,h_sh,N*4,hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_val,d_sh,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*8,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        int64_t exp=h_val[i] >> (h_sh[i] & 63);
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %lld exp %lld\n",i,(long long)h_out[i],(long long)exp);pass=0;break;}
    }
    if(pass)printf("out[1]=%lld out[%d]=%lld\n",(long long)h_out[1],N-1,(long long)h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_val));GPU_CHECK(hipFree(d_sh));GPU_CHECK(hipFree(d_out));
    free(h_val);free(h_sh);free(h_out);
    return pass?0:1;
}
