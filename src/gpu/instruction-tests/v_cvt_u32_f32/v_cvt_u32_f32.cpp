/*
 * HIP test for V_CVT_U32_F32 (VOP1, convert f32 -> u32, truncation).
 * gem5 (vop1.cc): for finite in-range non-negative S0, D.u32 = (u32)S0.f32
 * (truncate toward zero). NaN -> 0; +Inf/overflow -> UINT_MAX; negative -> 0.
 * This test uses only finite non-negative inputs below 2^31 so the reference
 * is a plain C truncating cast.
 *
 * Exercises base / dpp (quad_perm:[1,0,3,2]) / sdwa (src0_sel:DWORD) encodings.
 * Pattern A (vector), BLOCK_SIZE=64 = one wavefront per block.
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
kern_base(const float *a, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = a[i]; uint32_t d;
    __asm__ volatile("v_cvt_u32_f32_e32 %0, %1\n\t" : "=v"(d) : "v"(s));
    out[i] = d;
}

__global__ void
kern_dpp(const float *a, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = a[i]; uint32_t d;
    __asm__ volatile(
        "v_cvt_u32_f32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t"
        : "=v"(d) : "v"(s));
    out[i] = d;
}

__global__ void
kern_sdwa(const float *a, uint32_t *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = a[i]; uint32_t d;
    __asm__ volatile(
        "v_cvt_u32_f32_sdwa %0, %1 dst_sel:DWORD dst_unused:UNUSED_PRESERVE "
        "src0_sel:DWORD\n\t"
        : "=v"(d) : "v"(s));
    out[i] = d;
}

int main(void)
{
    size_t szf = N * sizeof(float), szu = N * sizeof(uint32_t);
    float *h_a=(float*)malloc(szf);
    uint32_t *h_b=(uint32_t*)malloc(szu),*h_d=(uint32_t*)malloc(szu),
             *h_s=(uint32_t*)malloc(szu);
    for (int i = 0; i < N; i++) h_a[i] = (float)(i * 131 % 100000) + 0.75f;

    float *d_a; uint32_t *d_base,*d_dpp,*d_sdwa;
    GPU_CHECK(hipMalloc(&d_a,szf));GPU_CHECK(hipMalloc(&d_base,szu));
    GPU_CHECK(hipMalloc(&d_dpp,szu));GPU_CHECK(hipMalloc(&d_sdwa,szu));
    GPU_CHECK(hipMemcpy(d_a,h_a,szf,hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    kern_base<<<grid,BLOCK_SIZE>>>(d_a,d_base,N);
    kern_dpp <<<grid,BLOCK_SIZE>>>(d_a,d_dpp ,N);
    kern_sdwa<<<grid,BLOCK_SIZE>>>(d_a,d_sdwa,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_b,d_base,szu,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_d,d_dpp ,szu,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(h_s,d_sdwa,szu,hipMemcpyDeviceToHost));

    const int perm[4] = {1,0,3,2};
    int pass = 1;
    for (int i = 0; i < N && pass; i++) {
        uint32_t eb = (uint32_t)h_a[i];
        if (h_b[i] != eb) { fprintf(stderr,"BASE FAIL[%d] got %u exp %u\n",i,h_b[i],eb);pass=0;break; }
        int lane = i % BLOCK_SIZE, base = i - lane;
        int src = base + ((lane & ~3) + perm[lane & 3]);
        uint32_t ed = (uint32_t)h_a[src];
        if (h_d[i] != ed) { fprintf(stderr,"DPP FAIL[%d] got %u exp %u\n",i,h_d[i],ed);pass=0;break; }
        if (h_s[i] != eb) { fprintf(stderr,"SDWA FAIL[%d] got %u exp %u\n",i,h_s[i],eb);pass=0;break; }
    }
    if (pass) printf("base[4]=%u dpp[4]=%u sdwa[4]=%u\n",h_b[4],h_d[4],h_s[4]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_a));GPU_CHECK(hipFree(d_base));
    GPU_CHECK(hipFree(d_dpp));GPU_CHECK(hipFree(d_sdwa));
    free(h_a);free(h_b);free(h_d);free(h_s);
    return pass ? 0 : 1;
}
