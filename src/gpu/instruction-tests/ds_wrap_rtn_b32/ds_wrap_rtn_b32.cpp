/*
 * HIP test for DS_WRAP_RTN_B32 (ENC_DS, LDS wrap atomic with return),
 * pattern C.
 *
 * gem5 (src/arch/amdgpu/common/insts/ds.cc + base/amo.hh AtomicOpWrap):
 *   old = MEM[ADDR]
 *   MEM[ADDR] = (old >= DATA0) ? old - DATA0 : old + DATA1
 *   RETURN_DATA = old
 *
 * Each lane owns a private LDS dword.  We seed it with a known value, run the
 * wrap atomic with per-lane DATA0/DATA1 chosen to exercise both branches, then
 * verify both the returned old value and the new stored value.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64;

__global__ void kernel(const uint32_t*m,const uint32_t*a,const uint32_t*b,
                       uint32_t*ret,uint32_t*newv,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    __shared__ uint32_t lds[BLOCK_SIZE];
    int t=threadIdx.x;
    lds[t]=m[i];
    __syncthreads();
    __attribute__((address_space(3))) uint32_t*lp=(__attribute__((address_space(3))) uint32_t*)(&lds[t]);
    uint32_t addr=(uint32_t)(uintptr_t)lp;
    uint32_t data0=a[i], data1=b[i], old;
    __asm__ volatile(
        "ds_wrap_rtn_b32 %0, %1, %2, %3\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"=v"(old):"v"(addr),"v"(data0),"v"(data1):"memory");
    ret[i]=old;
    __syncthreads();
    newv[i]=lds[t];
}
int main(void){
    uint32_t*h_m=(uint32_t*)malloc(N*4),*h_a=(uint32_t*)malloc(N*4),*h_b=(uint32_t*)malloc(N*4);
    uint32_t*h_ret=(uint32_t*)malloc(N*4),*h_new=(uint32_t*)malloc(N*4);
    for(int i=0;i<N;i++){
        h_m[i]=(uint32_t)(i*97u+5u);
        h_a[i]=(i&1)?(h_m[i]+10u):(h_m[i]-3u);   /* alternate old<a and old>=a */
        h_b[i]=(uint32_t)(i*7u+1u);
    }
    uint32_t*d_m,*d_a,*d_b,*d_ret,*d_new;
    GPU_CHECK(hipMalloc(&d_m,N*4));GPU_CHECK(hipMalloc(&d_a,N*4));GPU_CHECK(hipMalloc(&d_b,N*4));
    GPU_CHECK(hipMalloc(&d_ret,N*4));GPU_CHECK(hipMalloc(&d_new,N*4));
    GPU_CHECK(hipMemcpy(d_m,h_m,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*4,hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_m,d_a,d_b,d_ret,d_new,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_ret,d_ret,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_new,d_new,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t old=h_m[i];
        uint32_t exp=(old>=h_a[i])?(old-h_a[i]):(old+h_b[i]);
        if(h_ret[i]!=old||h_new[i]!=exp){
            fprintf(stderr,"FAIL[%d] ret=%u(e %u) new=%u(e %u)\n",i,h_ret[i],old,h_new[i],exp);pass=0;break;}}
    if(pass)printf("ret[1]=%u new[1]=%u\n",h_ret[1],h_new[1]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
