#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
static const int BLOCK_SIZE=64;

/* S_SET_GPR_IDX_ON (ENC_SOPC): enables GPR index mode (M0 index).
 * gem5: scoreboard_check_stage.cc:241 's_set_gpr_idx_on is of unknown type'. */
__global__ void kernel(int*out){
    uint32_t idx=0;
    __asm__ volatile("s_set_gpr_idx_on %0, gpr_idx(SRC0)\n\t"
                     "s_set_gpr_idx_off\n\t"::"s"(idx):"memory");
    if(threadIdx.x==0) out[0]=1;
}
int main(void){
    int h=0,*d; GPU_CHECK(hipMalloc(&d,4)); GPU_CHECK(hipMemcpy(d,&h,4,hipMemcpyHostToDevice));
    kernel<<<1,BLOCK_SIZE>>>(d);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(&h,d,4,hipMemcpyDeviceToHost));
    int pass=(h==1);
    printf("out=%d\n",h); printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
