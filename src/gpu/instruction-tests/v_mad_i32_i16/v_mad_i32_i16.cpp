/*
 * HIP test for V_MAD_I32_I16 (ENC_VOP3, signed 16x16+32 multiply-add).
 *
 *   D.i32 = sext(S0.i16) * sext(S1.i16) + S2.i32
 *
 * S0 and S1 are read as signed 16-bit values (low half of the VGPR, sign
 * extended); S2 and the destination are 32-bit.  Pattern A (vector), base
 * only.  Inputs span both signs to exercise sign extension.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_mad_i32_i16 v_mad_i32_i16.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                          \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N=256, BLOCK_SIZE=64;

__global__ void
kernel(const int16_t *a, const int16_t *b, const int32_t *c, int32_t *o, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* Sign-extend the 16-bit inputs into a 32-bit VGPR; the instruction reads
     * only the low 16 bits and sign-extends them itself. */
    int32_t va = (int32_t)a[i], vb = (int32_t)b[i], vc = c[i], vd;
    __asm__ volatile(
        "v_mad_i32_i16 %0, %1, %2, %3\n\t"
        : "=v"(vd) : "v"(va), "v"(vb), "v"(vc));
    o[i] = vd;
}

int
main(void)
{
    int16_t *h_a=(int16_t*)malloc(N*sizeof(int16_t));
    int16_t *h_b=(int16_t*)malloc(N*sizeof(int16_t));
    int32_t *h_c=(int32_t*)malloc(N*sizeof(int32_t));
    int32_t *h_o=(int32_t*)malloc(N*sizeof(int32_t));
    for (int i=0;i<N;i++){
        h_a[i]=(int16_t)((i*37+11)%65536 - 32768);
        h_b[i]=(int16_t)((i*53+7)%65536 - 32768);
        h_c[i]=(int32_t)(i*1234567 - 100000);
    }
    int16_t *d_a,*d_b; int32_t *d_c,*d_o;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(int16_t)));
    GPU_CHECK(hipMalloc(&d_b,N*sizeof(int16_t)));
    GPU_CHECK(hipMalloc(&d_c,N*sizeof(int32_t)));
    GPU_CHECK(hipMalloc(&d_o,N*sizeof(int32_t)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(int16_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(int16_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_c,h_c,N*sizeof(int32_t),hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_a,d_b,d_c,d_o,N);
    GPU_CHECK(hipGetLastError()); GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_o,d_o,N*sizeof(int32_t),hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++){
        int32_t exp=(int32_t)((int32_t)h_a[i]*(int32_t)h_b[i]+h_c[i]);
        if (h_o[i]!=exp){fprintf(stderr,"FAIL[%d] got %d exp %d\n",i,h_o[i],exp);pass=0;break;}
    }
    if (pass) printf("out[0]=%d out[%d]=%d\n",h_o[0],N-1,h_o[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
