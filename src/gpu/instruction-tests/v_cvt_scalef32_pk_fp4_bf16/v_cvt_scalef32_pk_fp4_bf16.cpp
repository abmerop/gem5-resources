/*
 * HIP test for V_CVT_SCALEF32_PK_FP4_BF16 (VOP3, MXFP scaled pack: two bf16
 * inputs -> two packed fp4 (e2m1) codes, scaled by an f32).
 *
 * gem5 model (Inst_VOP3__V_CVT_SCALE_PK in
 * src/arch/amdgpu/common/insts/vop3_cvt.hh):
 *   src0 holds two packed bf16 values (low half, high half).  Each is divided
 *   by the f32 scale (src1) and converted to fp4_e2m1; the two 4-bit codes are
 *   packed as (code(high) << 4) | code(low) into the low byte of vdst, leaving
 *   the remaining vdst bits unchanged.
 *
 * Verification: both bf16 halves are set to the same exactly-representable
 * value from an fp4 value/code table, with scale 1.0 (identity), so the
 * expected low byte is (code<<4)|code and the upper 24 bits stay 0 (dst
 * preloaded to 0, used read-write with "+v").  Pattern A, base only (no
 * DPP/SDWA for this VOP3-only op).
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define GPU_CHECK(cmd)                                                      \
    do { hipError_t e=(cmd); if(e!=hipSuccess){                            \
        fprintf(stderr,"HIP error %s:%d '%s'\n",__FILE__,__LINE__,         \
        hipGetErrorString(e)); exit(1);} } while(0)

static const int N=256, BLOCK_SIZE=64;

/* src0 = two packed bf16 (both equal); dst low byte = (code<<4)|code. */
__global__ void
kernel_cvt(const uint32_t *src, uint32_t *out, int n)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    uint32_t s = src[i];        /* two packed bf16 */
    float scale = 1.0f;
    uint32_t dst = 0;
    __asm__ volatile(
        "v_cvt_scalef32_pk_fp4_bf16 %0, %1, %2\n\t"
        : "+v"(dst)
        : "v"(s), "v"(scale));
    out[i] = dst;
}

struct VC { float v; uint8_t code; };

static uint16_t to_bf16(float f) { uint32_t b; __builtin_memcpy(&b,&f,4); return (uint16_t)(b>>16); }

int main(void)
{
    static const VC tbl[] = {
        {0.0f,0},{0.5f,1},{1.0f,2},{1.5f,3},{2.0f,4},{3.0f,5},{4.0f,6},{6.0f,7},
        {-0.5f,9},{-1.0f,10},{-1.5f,11},{-2.0f,12},{-3.0f,13},{-4.0f,14},{-6.0f,15},
    };
    const int T=(int)(sizeof(tbl)/sizeof(tbl[0]));

    uint32_t *h_src=(uint32_t*)malloc(N*4),*h_out=(uint32_t*)malloc(N*4);
    uint8_t *exp_code=(uint8_t*)malloc(N);
    for(int i=0;i<N;i++){
        VC e=tbl[i%T]; uint16_t bf=to_bf16(e.v);
        h_src[i]=((uint32_t)bf<<16)|(uint32_t)bf;   /* both halves same */
        exp_code[i]=e.code;
    }
    uint32_t *d_src,*d_out;
    GPU_CHECK(hipMalloc(&d_src,N*4));GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_src,h_src,N*4,hipMemcpyHostToDevice));
    kernel_cvt<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(d_src,d_out,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++){
        uint32_t expected=((uint32_t)exp_code[i]<<4)|(uint32_t)exp_code[i];
        if((h_out[i]&0xff)!=expected || (h_out[i]>>8)!=0){
            fprintf(stderr,"FAIL out[%d]: got 0x%08x exp low 0x%02x (v=%g)\n",
                    i,h_out[i],expected,tbl[i%T].v);pass=0;break;}
    }
    if(pass)printf("out[0]=0x%08x out[1]=0x%08x out[2]=0x%08x\n",h_out[0],h_out[1],h_out[2]);
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
