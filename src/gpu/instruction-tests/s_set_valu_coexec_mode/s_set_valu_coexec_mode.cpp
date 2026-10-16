/*
 * S_SET_VALU_COEXEC_MODE (ENC_SOPP) — NOT TESTABLE on gfx950 (CDNA4).
 *
 * This is an RDNA/gfx12 VALU co-execution mode instruction. The gfx950
 * assembler (llvm-mc -mcpu=gfx950) rejects the mnemonic as "invalid
 * instruction", so it cannot be emitted from a CDNA4 HIP kernel. gem5's vega
 * decoder has a decode_OP_SOPP__S_SET_VALU_COEXEC_MODE stub but there is no
 * way to reach it from a gfx950 program.
 *
 * Recorded as a failure: cannot emit the target instruction on gfx950.
 */
int main(void){ return 1; }
