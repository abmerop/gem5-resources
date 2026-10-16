/*
 * HIP test for V_CMPX_GT_I64 (EXEC/result bit = (S0.i64 > S1.i64); EXEC saved/restored).
 * gem5: src/arch/amdgpu/common/insts/vop3_cmp.cc. The _e64 form writes the
 * per-lane result to an SGPR mask; bit (threadIdx.x) is the lane's result.
 * For V_CMPX the EXEC mask is saved before and restored after.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                          \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern(const int64_t *a, const int64_t *b, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t A_ = a[i], B_ = b[i];
    int64_t va = A_;
    int64_t vb = B_;
    uint64_t mask, saved;
    __asm__ volatile(
        "s_mov_b64 %1, exec\n\t"
        "v_cmpx_gt_i64_e64 %0, %2, %3\n\t"
        "s_mov_b64 exec, %1\n\t"
        : "=s"(mask), "=s"(saved)
        : "v"(va), "v"(vb));
    out[i] = (int)((mask >> threadIdx.x) & 1ULL);
}

int main(void)
{
    int64_t *h_a=(int64_t*)malloc(N*sizeof(int64_t));
    int64_t *h_b=(int64_t*)malloc(N*sizeof(int64_t));
    int *h_out=(int*)malloc(N*sizeof(int));
    for(int i=0;i<N;i++){ h_a[i]=(int64_t)((int64_t)(i-128)*0x100000001ll); h_b[i]=(int64_t)((int64_t)(64-i)*0x100000001ll); }
    int64_t *d_a,*d_b; int *d_out;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(int64_t)));
    GPU_CHECK(hipMalloc(&d_b,N*sizeof(int64_t)));
    GPU_CHECK(hipMalloc(&d_out,N*sizeof(int)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(int64_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(int64_t),hipMemcpyHostToDevice));
    kern<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int),hipMemcpyDeviceToHost));
    int pass=1,ntrue=0;
    for(int i=0;i<N;i++){
        int64_t A=h_a[i], B=h_b[i];
        int exp=(A > B)?1:0; ntrue+=exp;
        if(h_out[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_out[i],exp);pass=0;break;}
    }
    if(pass)printf("true=%d/%d\n",ntrue,N);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_b));GPU_CHECK(hipFree(d_out));
    free(h_a);free(h_b);free(h_out);
    return pass?0:1;
}
