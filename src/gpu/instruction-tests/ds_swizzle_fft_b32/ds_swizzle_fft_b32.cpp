/*
 * HIP test for DS_SWIZZLE_B32 FFT and ROTATE modes (ENC_DS, cross-lane
 * swizzle, no LDS access), plus a QUAD_PERM regression check.
 *
 * gem5 ds.cc decodes the 16-bit ds_pattern (offset1:offset0) as:
 *   offset >= 0xe000 : FFT mode
 *                      j  = reverse_bits5(i & 0x1f) >> popcount(mask);
 *                      j |= (i & mask); j |= (i & 0x20);   mask = offset[4:0]
 *   offset >= 0xc000 : ROTATE mode
 *                      rot = offset[9:5] (negated if offset[10]);
 *                      j = (i & mask) | ((i + rot) & ~mask);
 *                      j = (j & 0x1f) | (i & 0x20);        mask = offset[4:0]
 *   offset[15] == 1  : QUAD_PERM (groups of 4)
 *   offset[15] == 0  : bitmask (and/or/xor over groups of 32)
 * A lane reading an inactive/invalid source lane returns 0.
 *
 * One wavefront (64 lanes) per block; all lanes active. The kernel applies
 * five patterns to the same input and the host recomputes each reference:
 *   0: QUAD_PERM(3,2,1,0)  offset 0x801b  (regression)
 *   1: FFT,0               offset 0xe000  (bit-reversal within 32)
 *   2: FFT,16              offset 0xe010
 *   3: ROTATE left 1       offset 0xc020
 *   4: ROTATE right 1      offset 0xc420
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int N=256,BLOCK_SIZE=64,NMODES=5;

__global__ void kernel(const uint32_t*in,uint32_t*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    uint32_t d=in[i],r0,r1,r2,r3,r4;
    __asm__ volatile("ds_swizzle_b32 %0, %1 offset:swizzle(QUAD_PERM,3,2,1,0)\n\ts_waitcnt lgkmcnt(0)":"=v"(r0):"v"(d));
    __asm__ volatile("ds_swizzle_b32 %0, %1 offset:swizzle(FFT,0)\n\ts_waitcnt lgkmcnt(0)":"=v"(r1):"v"(d));
    __asm__ volatile("ds_swizzle_b32 %0, %1 offset:swizzle(FFT,16)\n\ts_waitcnt lgkmcnt(0)":"=v"(r2):"v"(d));
    __asm__ volatile("ds_swizzle_b32 %0, %1 offset:swizzle(ROTATE,0,1)\n\ts_waitcnt lgkmcnt(0)":"=v"(r3):"v"(d));
    __asm__ volatile("ds_swizzle_b32 %0, %1 offset:swizzle(ROTATE,1,1)\n\ts_waitcnt lgkmcnt(0)":"=v"(r4):"v"(d));
    out[0*n+i]=r0; out[1*n+i]=r1; out[2*n+i]=r2; out[3*n+i]=r3; out[4*n+i]=r4;
}

static int rev5(int x){int r=0;for(int b=0;b<5;b++)r|=((x>>b)&1)<<(4-b);return r;}
static int popc(int x){int c=0;while(x){c+=x&1;x>>=1;}return c;}
// Source lane (within the 64-lane wavefront) for destination lane i.
static int src_quad(int i){int L=i&~3,k=i&3;return L+(3-k);}
static int src_fft(int i,int mask){int j=rev5(i&0x1f)>>popc(mask);j|=(i&mask);j|=(i&0x20);return j;}
static int src_rot(int i,int rot,int mask){int j=(i&mask)|((i+rot)&~mask);return (j&0x1f)|(i&0x20);}

int main(void){
    uint32_t*hi=(uint32_t*)malloc(N*4),*ho=(uint32_t*)malloc(N*NMODES*4);
    for(int i=0;i<N;i++) hi[i]=(uint32_t)(i*2654435761u+123u);
    uint32_t*di,*dO; GPU_CHECK(hipMalloc(&di,N*4));GPU_CHECK(hipMalloc(&dO,N*NMODES*4));
    GPU_CHECK(hipMemcpy(di,hi,N*4,hipMemcpyHostToDevice));
    kernel<<<N/BLOCK_SIZE,BLOCK_SIZE>>>(di,dO,N);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(ho,dO,N*NMODES*4,hipMemcpyDeviceToHost));
    const char*names[NMODES]={"QUAD_PERM","FFT,0","FFT,16","ROTATE_L1","ROTATE_R1"};
    int pass=1;
    for(int mode=0;mode<NMODES && pass;mode++){
        for(int i=0;i<N && pass;i++){
            int wbase=(i/BLOCK_SIZE)*BLOCK_SIZE, lane=i%BLOCK_SIZE, s;
            switch(mode){
                case 0: s=src_quad(lane); break;
                case 1: s=src_fft(lane,0); break;
                case 2: s=src_fft(lane,16); break;
                case 3: s=src_rot(lane,1,0); break;   // rotate left 1
                default:s=src_rot(lane,-1,0); break;  // rotate right 1
            }
            uint32_t exp=hi[wbase+s], got=ho[mode*N+i];
            if(got!=exp){fprintf(stderr,"FAIL %s[%d] lane%d got 0x%08x exp 0x%08x (src lane %d)\n",
                                  names[mode],i,lane,got,exp,s);pass=0;}
        }
    }
    if(pass){
        printf("FFT,0 out[0..3]=%08x %08x %08x %08x\n",ho[N+0],ho[N+1],ho[N+2],ho[N+3]);
        for(int m=0;m<NMODES;m++) printf("  %-10s OK\n",names[m]);
    }
    printf("%s\n",pass?"PASSED":"FAILED");return pass?0:1;
}
