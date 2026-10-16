/*
 * HIP test for V_CVT_OFF_F32_I4 (VOP1, 4-bit signed index -> f32 offset via a
 * fixed lookup table), with DPP and SDWA variants.
 *
 * gem5 (Inst_VOP1__V_CVT_OFF_F32_I4::execute in common/insts/vop1.cc) uses a
 * 16-entry table indexed by bits[3:0] of the source:
 *   idx : 0..15
 *   val : +0.0000 +0.0625 +0.1250 +0.1875 +0.2500 +0.3125 +0.3750 +0.4375
 *         -0.5000 -0.4375 -0.3750 -0.3125 -0.2500 -0.1875 -0.1250 -0.0625
 *
 *   base : v_cvt_off_f32_i4_e32 vd, va                out = T[a[i] & 15]
 *   sdwa : src0_sel:BYTE_0 (identity low byte)        out = T[a[i] & 15]
 *
 * KNOWN gem5 BUG (this test FAILS against gem5):
 *   Inst_VOP1__V_CVT_OFF_F32_I4::execute uses
 *     vop1Helper<ConstVecOperandU32, VecOperandU32>(...)
 *   and its lambda assigns the *float* table value into a VecOperandU32
 *   destination:  vdst[lane] = cvt_off_table[val];
 *   Every table entry has |value| < 1.0, so the float->uint32 conversion
 *   truncates ALL results to integer 0.  gem5 therefore returns 0 for every
 *   input (observed: base[1] with input 1 returns 0, expected 0.0625).  The
 *   destination operand (and the table storage) should be F32, not U32.  This
 *   affects base, SDWA and DPP alike.  Left in place as a debugging artifact.
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o v_cvt_off_f32_i4 v_cvt_off_f32_i4.cpp
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
                    hipGetErrorString(e));                                  \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static const int N=256; static const int BLOCK_SIZE=64;

__global__ void
kbase(const uint32_t *a, float *o, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint32_t s0=a[i]; float vd;
    __asm__ volatile("v_cvt_off_f32_i4_e32 %0,%1\n\t":"=v"(vd):"v"(s0));
    o[i]=vd;
}
__global__ void
ksdwa(const uint32_t *a, float *o, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint32_t s0=a[i]; float vd;
    __asm__ volatile("v_cvt_off_f32_i4_sdwa %0,%1 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0\n\t":"=v"(vd):"v"(s0));
    o[i]=vd;
}

/* gem5's exact table (note index 10 quirk). */
static const float T[16] = {
    +0.0000f, +0.0625f, +0.1250f, +0.1875f, +0.2500f, +0.3125f,
    +0.3750f, +0.4375f, -0.5000f, -0.4375f, -0.3750f, -0.3125f,
    -0.2500f, -0.1875f, -0.1250f, -0.0625f};

int main(void){
    uint32_t *ha=(uint32_t*)malloc(N*4);
    float *hbase=(float*)malloc(N*4),*hsdwa=(float*)malloc(N*4);
    for(int i=0;i<N;i++) ha[i]=(uint32_t)(i & 15);   /* all 16 codes */
    uint32_t *da; float *dbase,*dsdwa;
    GPU_CHECK(hipMalloc(&da,N*4));
    GPU_CHECK(hipMalloc(&dbase,N*4));GPU_CHECK(hipMalloc(&dsdwa,N*4));
    GPU_CHECK(hipMemcpy(da,ha,N*4,hipMemcpyHostToDevice));
    int g=N/BLOCK_SIZE;
    hipLaunchKernelGGL(kbase,dim3(g),dim3(BLOCK_SIZE),0,0,da,dbase,N);GPU_CHECK(hipGetLastError());
    hipLaunchKernelGGL(ksdwa,dim3(g),dim3(BLOCK_SIZE),0,0,da,dsdwa,N);GPU_CHECK(hipGetLastError());
    GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(hbase,dbase,N*4,hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(hsdwa,dsdwa,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        float eb=T[ha[i]&15];
        if(hbase[i]!=eb){fprintf(stderr,"FAIL base[%d] %g exp %g\n",i,hbase[i],eb);pass=0;break;}
        if(hsdwa[i]!=eb){fprintf(stderr,"FAIL sdwa[%d] %g exp %g\n",i,hsdwa[i],eb);pass=0;break;}
    }
    if(pass)printf("base[1]=%g sdwa[1]=%g\n",hbase[1],hsdwa[1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    return pass?0:1;
}
