/*
 * HIP test for DS_READ_B96_TR_B6 (ENC_DS) on CDNA4 (gfx950).  Pattern C (LDS).
 *
 *   VDST[0:2] = transpose6( LDS[ addr ] )   (96-bit transposing LDS read of
 *   sixteen 6-bit elements, transposed within each group of 16 lanes)
 *
 * The exact 6-bit transpose layout is complex, so this test performs the
 * minimal functional self-check that reliably distinguishes correct HW from
 * the known gem5 gap: LDS is seeded (from a global input the compiler cannot
 * elide) so that *every* 6-bit element of every lane is non-zero.  Under any
 * valid transpose permutation each result 6-bit element is therefore non-zero,
 * so every one of the three destination dwords must be non-zero.  The three
 * destination VGPRs are pre-loaded with a distinctive sentinel; if gem5 never
 * fills vdst (the historical gap) the result is all-zero and the test FAILS.
 *
 * NOTE on addressing/DCE: the ds_read reads its data from LDS, but that
 * dependency is invisible to the compiler.  We must (a) make the LDS address
 * genuinely escape into the asm (via an address_space(3) pointer, not a
 * base-relative offset that folds to pure arithmetic) and (b) add a "memory"
 * clobber, otherwise the compiler dead-store-eliminates the LDS initialization
 * and shrinks the group segment to 0 bytes, so the read returns undefined data.
 *
 * Base encoding only (DS has no DPP/SDWA).
 *
 * Build: hipcc -O2 --offload-arch=gfx950 -o ds_read_b96_tr_b6 ds_read_b96_tr_b6.cpp
 */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define GPU_CHECK(cmd) do{hipError_t e=(cmd);if(e!=hipSuccess){fprintf(stderr,    "HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e));exit(1);}}while(0)
static const int BLOCK_SIZE=64;
static const uint32_t SENT=0xDEADBEEFu;

__global__ void kernel(const uint32_t*data,uint32_t*out){
    __shared__ uint32_t lds[BLOCK_SIZE*3];
    int t=threadIdx.x;
    // Seed LDS from a global input so the stores cannot be optimized away.
    lds[t*3+0]=data[t*3+0];
    lds[t*3+1]=data[t*3+1];
    lds[t*3+2]=data[t*3+2];
    __syncthreads();

    // Derive the LDS byte address from a real LDS pointer so the shared array
    // genuinely escapes into the asm (prevents dead-store elimination).
    __attribute__((address_space(3))) uint32_t *lp =
        (__attribute__((address_space(3))) uint32_t *)(&lds[t*3]);
    uint32_t addr=(uint32_t)(uintptr_t)lp;
    typedef uint32_t u32x3 __attribute__((ext_vector_type(3)));
    u32x3 d; d[0]=SENT; d[1]=SENT; d[2]=SENT;
    __asm__ volatile(
        "ds_read_b96_tr_b6 %0, %1\n\t"
        "s_waitcnt lgkmcnt(0)\n\t"
        :"=v"(d):"v"(addr):"memory");
    out[t*3+0]=d[0]; out[t*3+1]=d[1]; out[t*3+2]=d[2];
}

// Pack 16 six-bit elements (LSB-first, element k at bits [6k+5:6k]) into a
// 96-bit value stored as three little-endian dwords.
static void pack96(uint32_t elems[16], uint32_t dw[3]){
    dw[0]=dw[1]=dw[2]=0;
    for(int k=0;k<16;k++){
        uint32_t v=elems[k]&0x3fu;
        int lo=6*k, lodw=lo/32, hidw=(lo+5)/32, off=lo%32;
        if(lodw==hidw){
            dw[lodw]|=v<<off;
        }else{
            int loBits=32-off;
            dw[lodw]|=(v&((1u<<loBits)-1))<<off;
            dw[hidw]|=v>>loBits;
        }
    }
}

int main(void){
    const int N=BLOCK_SIZE*3;
    uint32_t*h_data=(uint32_t*)malloc(N*4);
    uint32_t*h_out=(uint32_t*)malloc(N*4);
    // Build per-lane LDS data whose every 6-bit element is non-zero (in [1,63]).
    for(int t=0;t<BLOCK_SIZE;t++){
        uint32_t elems[16], dw[3];
        for(int k=0;k<16;k++) elems[k]=((t+k)&0x3fu)|1u; // odd => never zero
        pack96(elems,dw);
        h_data[t*3+0]=dw[0]; h_data[t*3+1]=dw[1]; h_data[t*3+2]=dw[2];
    }
    uint32_t*d_data,*d_out;
    GPU_CHECK(hipMalloc(&d_data,N*4)); GPU_CHECK(hipMalloc(&d_out,N*4));
    GPU_CHECK(hipMemcpy(d_data,h_data,N*4,hipMemcpyHostToDevice));
    kernel<<<1,BLOCK_SIZE>>>(d_data,d_out);
    GPU_CHECK(hipGetLastError());GPU_CHECK(hipDeviceSynchronize());
    GPU_CHECK(hipMemcpy(h_out,d_out,N*4,hipMemcpyDeviceToHost));
    int pass=1;
    for(int i=0;i<N;i++) if(h_out[i]==SENT||h_out[i]==0u){
        fprintf(stderr,"FAIL out[%d]=0x%08x (expected real non-zero LDS data)\n",i,h_out[i]);pass=0;break;}
    if(pass) printf("out[0]=0x%08x out[%d]=0x%08x\n",h_out[0],N-1,h_out[N-1]);
    printf("%s\n",pass?"PASSED":"FAILED");
    GPU_CHECK(hipFree(d_data));GPU_CHECK(hipFree(d_out));free(h_data);free(h_out);
    return pass?0:1;
}
