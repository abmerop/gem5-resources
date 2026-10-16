/* HIP test for S_ATOMIC_UMAX (SMEM 32-bit scalar atomic, gfx950). One
 * wavefront per counter; old value returned via glc. base only. */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int NUM_WF=64;
__global__ void kernel(uint32_t*counters,uint32_t*old_out,uint32_t dval){
    int wf=blockIdx.x; uint32_t*slot=&counters[wf];
    uint32_t old=dval;
    __asm__ volatile("s_atomic_umax %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)\n\t":"+s"(old):"s"(slot):"memory");
    if(threadIdx.x==0) old_out[wf]=old;
}
int main(void){
    uint32_t h_init[NUM_WF],h_mem[NUM_WF],h_old[NUM_WF];
    const uint32_t DVAL=(uint32_t)(2147483648);
    for(int w=0;w<NUM_WF;w++) h_init[w]=(uint32_t)(w*2654435761u+123u);
    uint32_t*d_c,*d_old;
    GPU_CHECK(hipMalloc(&d_c,NUM_WF*sizeof(uint32_t)));GPU_CHECK(hipMalloc(&d_old,NUM_WF*sizeof(uint32_t)));
    GPU_CHECK(hipMemcpy(d_c,h_init,NUM_WF*sizeof(uint32_t),hipMemcpyHostToDevice));
    kernel<<<NUM_WF,64>>>(d_c,d_old,DVAL);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_mem,d_c,NUM_WF*sizeof(uint32_t),hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_old,d_old,NUM_WF*sizeof(uint32_t),hipMemcpyDeviceToHost));
    int pass=1;
    for(int w=0;w<NUM_WF;w++){
        uint32_t a=h_init[w],b=DVAL; uint32_t em=(a>b?a:b); uint32_t eo=a;
        if(h_old[w]!=eo||h_mem[w]!=em){
            fprintf(stderr,"FAIL[%d] old=%u(e %u) mem=%u(e %u)\n",w,h_old[w],eo,h_mem[w],em);pass=0;break;}
    }
    if(pass)printf("old[0]=%u mem[0]=%u\n",h_old[0],h_mem[0]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
