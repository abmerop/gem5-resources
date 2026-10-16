/*
 * HIP test for V_MAX_I32 (VOP2 signed 32-bit maximum).
 *
 * gem5 (src/arch/amdgpu/common/insts/) computes: out = a > b ? a : b
 * Inputs are exact integers so the result is bit-exact.  This instruction's
 * SDWA/DPP encodings hit a panic_if guard in gem5 (not implemented), so only
 * the base (e32) encoding is exercised here -- variants stay not-working in
 * the DB, matching gem5 support (cf. v_trunc_f32 precedent).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_max_i32 v_max_i32.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__, \
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N = 256;
static const int BLOCK_SIZE = 64;

__global__ void
kernel_v_max_i32(const int32_t *a, const int32_t *b, int32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t va = (uint32_t)a[i], vb = (uint32_t)b[i], vd;
    __asm__ volatile("v_max_i32_e32 %0, %1, %2\n\t"
                     : "=v"(vd) : "v"(va), "v"(vb));
    out[i] = (int32_t)vd;
}

int
main(void)
{
    int32_t *h_a=(int32_t*)malloc(N*sizeof(int32_t));
    int32_t *h_b=(int32_t*)malloc(N*sizeof(int32_t));
    int32_t *h_out=(int32_t*)malloc(N*sizeof(int32_t));
    for (int i=0;i<N;i++) { int32_t a=(int32_t)(i-128)*7; int32_t b=(int32_t)(64-i)*5; h_a[i]=a; h_b[i]=b; }
    int32_t *d_a,*d_b,*d_out;
    GPU_CHECK(hipMalloc(&d_a,N*sizeof(int32_t)));
    GPU_CHECK(hipMalloc(&d_b,N*sizeof(int32_t)));
    GPU_CHECK(hipMalloc(&d_out,N*sizeof(int32_t)));
    GPU_CHECK(hipMemcpy(d_a,h_a,N*sizeof(int32_t),hipMemcpyHostToDevice));
    GPU_CHECK(hipMemcpy(d_b,h_b,N*sizeof(int32_t),hipMemcpyHostToDevice));
    hipLaunchKernelGGL(kernel_v_max_i32,dim3(N/BLOCK_SIZE),dim3(BLOCK_SIZE),0,0,d_a,d_b,d_out,N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*sizeof(int32_t),hipMemcpyDeviceToHost));
    int pass=1;
    for (int i=0;i<N;i++) {
        int32_t a=h_a[i], b=h_b[i];
        int32_t expected = (int32_t)(a > b ? a : b);
        if (h_out[i]!=expected) {
            fprintf(stderr,"FAIL[%d] got %lld exp %lld\n",i,(long long)h_out[i],(long long)expected);
            pass=0; break;
        }
    }
    if (pass) printf("out[0]=%lld out[255]=%lld\n",(long long)h_out[0],(long long)h_out[255]);
    printf("%s\n", pass?"PASSED":"FAILED");
    return pass?0:1;
}
