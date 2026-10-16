/*
 * HIP test for SCRATCH_LOAD_LDS_SBYTE (ENC_FLAT scratch segment, direct
 * scratch/private -> LDS DMA load of a signed byte, sign-extended to a
 * 32-bit LDS dword).
 *
 * In gem5 this decodes onto Inst_FLAT__FLAT_LOAD_LDS_SBYTE with the scratch
 * (private) segment selected (isFlatScratch()); the SCRATCH_LOAD_LDS_* opcodes
 * share the FLAT_LOAD_LDS_* implementations (see vega/gpu_decoder.cc). This is
 * an LDS-DMA load: there is NO VGPR destination. Each active lane reads one
 * byte from its private/scratch aperture and the hardware writes it, sign-
 * extended to 32 bits, into LDS:
 *
 *   src (per lane, ST form "off, off"): flat_scratch + swizzleAddr(inst_off,lane)
 *   dst (per lane): lds_addr = m0[17:2]*4 + inst_off, then + lane*1*4
 *   value:          sext_8_to_32( scratch_byte )
 *
 * (see gem5 vega/insts/op_encodings.hh calcAddr(scratch) + ldsComplete<1,8,1>
 * and common/insts/flat.cc Inst_FLAT__FLAT_LOAD_LDS_SBYTE::completeAcc.)
 *
 * We use inst_off = 0 for both the store and the LDS-DMA so the single OFFSET
 * field is reused consistently: the scratch source byte and the LDS dword both
 * anchor at offset 0, and the LDS destination is m0[17:2]*4 (+ lane*4).
 *
 * Verification is a per-lane round trip entirely in gem5's own memories:
 *   1. scratch_store_byte writes a known (possibly negative) byte into this
 *      lane's private scratch slot (ST form, inst offset 0).
 *   2. scratch_load_lds_sbyte DMAs that byte, sign-extended, into lds[lane].
 *   3. after s_waitcnt + __syncthreads, out[i] = lds[threadIdx.x].
 * The host reference is simply the sign-extended input byte.
 *
 * The compiler never emits scratch_load_lds_sbyte on its own, so inline asm is
 * required. lds[] is the kernel's only shared allocation, so it sits at LDS
 * offset 0; m0 is set to its real byte offset (0) and &lds[0] is passed into
 * the asm so LLVM keeps the group segment allocated and does not DCE the read.
 * A volatile private array forces a scratch (private) segment so flat_scratch
 * is valid.
 *
 * Build:
 *   make -C hip_tests/scratch_load_lds_sbyte AMDGPU_TARGET=gfx950
 *
 * Run in gem5:
 *   gem5.opt config.py -- ./scratch_load_lds_sbyte
 */

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do {                                                                    \
        hipError_t e = (cmd);                                               \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s:%d '%s'\n", __FILE__, __LINE__,  \
                    hipGetErrorString(e));                                   \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N          = 256;
static const int BLOCK_SIZE = 64;       /* one wavefront per block */

__global__ void
kernel_scratch_load_lds_sbyte(const int8_t *g, int *out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    __shared__ int lds[BLOCK_SIZE];     /* only shared alloc -> LDS offset 0 */
    int t = threadIdx.x;

    /*
     * Force a private/scratch segment so flat_scratch is valid.  The volatile
     * store/read keeps the array live; its bytes overlap the scratch region the
     * asm uses at inst offset 0 (same idiom as scratch_load_sbyte).
     */
    volatile int pad[8];
    pad[t & 7] = i;

    int val = (int)g[i];                /* signed byte payload (low 8 bits) */

    /* Real LDS byte offset of lds[]; passed to m0 and escaped into the asm. */
    uint32_t lds_off = (uint32_t)(uintptr_t)lds;

    __asm__ volatile(
        /* 1. Put the payload byte into this lane's private scratch slot. */
        "scratch_store_byte off, %1, off offset:0\n\t"
        "s_waitcnt vmcnt(0)\n\t"
        /* 2. Point m0 at lds[] and DMA the signed byte into LDS. */
        "s_mov_b32 m0, %2\n\t"
        "scratch_load_lds_sbyte off, off offset:0\n\t"
        "s_waitcnt vmcnt(0) lgkmcnt(0)\n\t"
        :
        : "v"(lds), "v"(val), "s"(lds_off)
        : "memory", "m0");

    __syncthreads();
    out[i] = lds[t];
}

int
main(void)
{
    int8_t *h_g   = (int8_t *)malloc(N * sizeof(int8_t));
    int    *h_out = (int    *)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++)
        h_g[i] = (int8_t)(i * 7 - 100);   /* mix of negative & positive */

    int8_t *d_g;
    int    *d_out;
    GPU_CHECK(hipMalloc(&d_g,   N * sizeof(int8_t)));
    GPU_CHECK(hipMalloc(&d_out, N * sizeof(int)));

    GPU_CHECK(hipMemcpy(d_g, h_g, N * sizeof(int8_t), hipMemcpyHostToDevice));

    int grid = N / BLOCK_SIZE;
    hipLaunchKernelGGL(kernel_scratch_load_lds_sbyte, dim3(grid),
                       dim3(BLOCK_SIZE), 0, 0, d_g, d_out, N);
    GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());

    GPU_CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    int pass = 1;
    for (int i = 0; i < N; i++) {
        int expected = (int)h_g[i];   /* sign-extended byte -> i32 */
        if (h_out[i] != expected) {
            fprintf(stderr, "FAIL out[%d]: got %d expected %d\n",
                    i, h_out[i], expected);
            pass = 0; break;
        }
    }

    if (pass)
        printf("out[0]=%d out[%d]=%d\n", h_out[0], N - 1, h_out[N - 1]);
    printf("%s\n", pass ? "PASSED" : "FAILED");

    GPU_CHECK(hipFree(d_g)); GPU_CHECK(hipFree(d_out));
    free(h_g); free(h_out);
    return pass ? 0 : 1;
}
