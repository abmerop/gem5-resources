/*
 * HIP test for V_SQRT_F32 (VOP1, single-precision square root).
 * gem5 (vop1.cc): D.f32 = std::sqrt(S0.f32), per active lane.
 *
 * Exercises three encodings of the same opcode:
 *   base : v_sqrt_f32_e32  vd, vs
 *   dpp  : v_sqrt_f32_dpp  vd, vs quad_perm:[1,0,3,2]  (cross-lane src select)
 *   sdwa : v_sqrt_f32_sdwa vd, vs src0_sel:DWORD       (whole-dword identity)
 *
 * gem5 applies the DPP lane permutation to the source before the op
 * (op_encodings.hh dppHelper/processDPP) and applies SDWA src select
 * likewise. quad_perm:[1,0,3,2] maps, within each group of 4 lanes,
 *   out[q*4+0]=in[q*4+1], out[q*4+1]=in[q*4+0],
 *   out[q*4+2]=in[q*4+3], out[q*4+3]=in[q*4+2].
 *
 * Pattern A (vector): per-lane "v" operands. BLOCK_SIZE=64 = one wavefront
 * per block so DPP quads line up with threadIdx.x.
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N = 256, BLOCK_SIZE = 64;

__global__ void
kern_base(const float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = a[i], d;
    __asm__ volatile("v_sqrt_f32_e32 %0, %1\n\t" : "=v"(d) : "v"(s));
    out[i] = d;
}

__global__ void
kern_dpp(const float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = a[i], d;
    __asm__ volatile(
        "v_sqrt_f32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t"
        : "=v"(d) : "v"(s));
    out[i] = d;
}

__global__ void
kern_sdwa(const float *a, float *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = a[i], d;
    __asm__ volatile(
        "v_sqrt_f32_sdwa %0, %1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE "
        "src0_sel:DWORD\n\t"
        : "=v"(d) : "v"(s));
    out[i] = d;
}

static int
close_f(float got, float exp)
{
    if (isnan(got) && isnan(exp)) return 1;
    float d = fabsf(got - exp);
    return d <= 1e-4f * (1.0f + fabsf(exp));
}

int main(void)
{
    size_t sz = N * sizeof(float);
    float *h_a=(float*)malloc(sz), *h_b=(float*)malloc(sz),
          *h_d=(float*)malloc(sz), *h_s=(float*)malloc(sz);
    for (int i = 0; i < N; i++) h_a[i] = (float)(i % 97) * 1.5f + 0.25f;

    float *d_a,*d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,sz));GPU_CHECK(hipMalloc(&d_base,sz));
    GPU_CHECK(hipMalloc(&d_dpp,sz));GPU_CHECK(hipMalloc(&d_sdwa,sz));
    GPU_CHECK(hipMemcpy(d_a,h_a,sz,hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    kern_base<<<grid,BLOCK_SIZE>>>(d_a,d_base,N);
    kern_dpp <<<grid,BLOCK_SIZE>>>(d_a,d_dpp ,N);
    kern_sdwa<<<grid,BLOCK_SIZE>>>(d_a,d_sdwa,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_b,d_base,sz,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_dpp ,sz,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_s,d_sdwa,sz,hipMemcpyDeviceToHost));

    const int perm[4] = {1,0,3,2};
    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        float eb = sqrtf(h_a[i]);
        if (!close_f(h_b[i], eb)) {
            fprintf(stderr,"BASE FAIL[%d] got %g exp %g\n",i,h_b[i],eb);pass=0;break;
        }
        int lane = i % BLOCK_SIZE, base = i - lane;
        int src = base + ((lane & ~3) + perm[lane & 3]);
        float ed = sqrtf(h_a[src]);
        if (!close_f(h_d[i], ed)) {
            fprintf(stderr,"DPP FAIL[%d] got %g exp %g\n",i,h_d[i],ed);pass=0;break;
        }
        if (!close_f(h_s[i], eb)) {
            fprintf(stderr,"SDWA FAIL[%d] got %g exp %g\n",i,h_s[i],eb);pass=0;break;
        }
    }
    if (pass) printf("base[4]=%g dpp[4]=%g sdwa[4]=%g\n",h_b[4],h_d[4],h_s[4]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_base));
    GPU_CHECK(hipFree(d_dpp));GPU_CHECK(hipFree(d_sdwa));
    free(h_a);free(h_b);free(h_d);free(h_s);
    return pass ? 0 : 1;
}
