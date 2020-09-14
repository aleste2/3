package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/timer"
	"sync"
	"unsafe"
)

// CUDA handle for LLBtorque4 kernel
var LLBtorque4_code cu.Function

// Stores the arguments for LLBtorque4 kernel invocation
type LLBtorque4_args_t struct {
	arg_tx         unsafe.Pointer
	arg_ty         unsafe.Pointer
	arg_tz         unsafe.Pointer
	arg_mx         unsafe.Pointer
	arg_my         unsafe.Pointer
	arg_mz         unsafe.Pointer
	arg_hx         unsafe.Pointer
	arg_hy         unsafe.Pointer
	arg_hz         unsafe.Pointer
	arg_alpha_     unsafe.Pointer
	arg_alpha_mul  float32
	arg_TCurie_    unsafe.Pointer
	arg_TCurie_mul float32
	arg_Msat_      unsafe.Pointer
	arg_Msat_mul   float32
	arg_hth1x      unsafe.Pointer
	arg_hth1y      unsafe.Pointer
	arg_hth1z      unsafe.Pointer
	arg_hth2x      unsafe.Pointer
	arg_hth2y      unsafe.Pointer
	arg_hth2z      unsafe.Pointer
	arg_temp_      unsafe.Pointer
	arg_temp_mul   float32
	arg_a1_        unsafe.Pointer
	arg_a1_mul     float32
	arg_N          int
	argptr         [26]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for LLBtorque4 kernel invocation
var LLBtorque4_args LLBtorque4_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	LLBtorque4_args.argptr[0] = unsafe.Pointer(&LLBtorque4_args.arg_tx)
	LLBtorque4_args.argptr[1] = unsafe.Pointer(&LLBtorque4_args.arg_ty)
	LLBtorque4_args.argptr[2] = unsafe.Pointer(&LLBtorque4_args.arg_tz)
	LLBtorque4_args.argptr[3] = unsafe.Pointer(&LLBtorque4_args.arg_mx)
	LLBtorque4_args.argptr[4] = unsafe.Pointer(&LLBtorque4_args.arg_my)
	LLBtorque4_args.argptr[5] = unsafe.Pointer(&LLBtorque4_args.arg_mz)
	LLBtorque4_args.argptr[6] = unsafe.Pointer(&LLBtorque4_args.arg_hx)
	LLBtorque4_args.argptr[7] = unsafe.Pointer(&LLBtorque4_args.arg_hy)
	LLBtorque4_args.argptr[8] = unsafe.Pointer(&LLBtorque4_args.arg_hz)
	LLBtorque4_args.argptr[9] = unsafe.Pointer(&LLBtorque4_args.arg_alpha_)
	LLBtorque4_args.argptr[10] = unsafe.Pointer(&LLBtorque4_args.arg_alpha_mul)
	LLBtorque4_args.argptr[11] = unsafe.Pointer(&LLBtorque4_args.arg_TCurie_)
	LLBtorque4_args.argptr[12] = unsafe.Pointer(&LLBtorque4_args.arg_TCurie_mul)
	LLBtorque4_args.argptr[13] = unsafe.Pointer(&LLBtorque4_args.arg_Msat_)
	LLBtorque4_args.argptr[14] = unsafe.Pointer(&LLBtorque4_args.arg_Msat_mul)
	LLBtorque4_args.argptr[15] = unsafe.Pointer(&LLBtorque4_args.arg_hth1x)
	LLBtorque4_args.argptr[16] = unsafe.Pointer(&LLBtorque4_args.arg_hth1y)
	LLBtorque4_args.argptr[17] = unsafe.Pointer(&LLBtorque4_args.arg_hth1z)
	LLBtorque4_args.argptr[18] = unsafe.Pointer(&LLBtorque4_args.arg_hth2x)
	LLBtorque4_args.argptr[19] = unsafe.Pointer(&LLBtorque4_args.arg_hth2y)
	LLBtorque4_args.argptr[20] = unsafe.Pointer(&LLBtorque4_args.arg_hth2z)
	LLBtorque4_args.argptr[21] = unsafe.Pointer(&LLBtorque4_args.arg_temp_)
	LLBtorque4_args.argptr[22] = unsafe.Pointer(&LLBtorque4_args.arg_temp_mul)
	LLBtorque4_args.argptr[23] = unsafe.Pointer(&LLBtorque4_args.arg_a1_)
	LLBtorque4_args.argptr[24] = unsafe.Pointer(&LLBtorque4_args.arg_a1_mul)
	LLBtorque4_args.argptr[25] = unsafe.Pointer(&LLBtorque4_args.arg_N)
}

// Wrapper for LLBtorque4 CUDA kernel, asynchronous.
func k_LLBtorque4_async(tx unsafe.Pointer, ty unsafe.Pointer, tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, hx unsafe.Pointer, hy unsafe.Pointer, hz unsafe.Pointer, alpha_ unsafe.Pointer, alpha_mul float32, TCurie_ unsafe.Pointer, TCurie_mul float32, Msat_ unsafe.Pointer, Msat_mul float32, hth1x unsafe.Pointer, hth1y unsafe.Pointer, hth1z unsafe.Pointer, hth2x unsafe.Pointer, hth2y unsafe.Pointer, hth2z unsafe.Pointer, temp_ unsafe.Pointer, temp_mul float32, a1_ unsafe.Pointer, a1_mul float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("LLBtorque4")
	}

	LLBtorque4_args.Lock()
	defer LLBtorque4_args.Unlock()

	if LLBtorque4_code == 0 {
		LLBtorque4_code = fatbinLoad(LLBtorque4_map, "LLBtorque4")
	}

	LLBtorque4_args.arg_tx = tx
	LLBtorque4_args.arg_ty = ty
	LLBtorque4_args.arg_tz = tz
	LLBtorque4_args.arg_mx = mx
	LLBtorque4_args.arg_my = my
	LLBtorque4_args.arg_mz = mz
	LLBtorque4_args.arg_hx = hx
	LLBtorque4_args.arg_hy = hy
	LLBtorque4_args.arg_hz = hz
	LLBtorque4_args.arg_alpha_ = alpha_
	LLBtorque4_args.arg_alpha_mul = alpha_mul
	LLBtorque4_args.arg_TCurie_ = TCurie_
	LLBtorque4_args.arg_TCurie_mul = TCurie_mul
	LLBtorque4_args.arg_Msat_ = Msat_
	LLBtorque4_args.arg_Msat_mul = Msat_mul
	LLBtorque4_args.arg_hth1x = hth1x
	LLBtorque4_args.arg_hth1y = hth1y
	LLBtorque4_args.arg_hth1z = hth1z
	LLBtorque4_args.arg_hth2x = hth2x
	LLBtorque4_args.arg_hth2y = hth2y
	LLBtorque4_args.arg_hth2z = hth2z
	LLBtorque4_args.arg_temp_ = temp_
	LLBtorque4_args.arg_temp_mul = temp_mul
	LLBtorque4_args.arg_a1_ = a1_
	LLBtorque4_args.arg_a1_mul = a1_mul
	LLBtorque4_args.arg_N = N

	args := LLBtorque4_args.argptr[:]
	cu.LaunchKernel(LLBtorque4_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("LLBtorque4")
	}
}

// maps compute capability on PTX code for LLBtorque4 kernel.
var LLBtorque4_map = map[int]string{0: "",
	30: LLBtorque4_ptx_30}

// LLBtorque4 PTX code for various compute capabilities.
const (
	LLBtorque4_ptx_30 = `
.version 6.5
.target sm_30
.address_size 64

	// .globl	LLBtorque4

.visible .entry LLBtorque4(
	.param .u64 LLBtorque4_param_0,
	.param .u64 LLBtorque4_param_1,
	.param .u64 LLBtorque4_param_2,
	.param .u64 LLBtorque4_param_3,
	.param .u64 LLBtorque4_param_4,
	.param .u64 LLBtorque4_param_5,
	.param .u64 LLBtorque4_param_6,
	.param .u64 LLBtorque4_param_7,
	.param .u64 LLBtorque4_param_8,
	.param .u64 LLBtorque4_param_9,
	.param .f32 LLBtorque4_param_10,
	.param .u64 LLBtorque4_param_11,
	.param .f32 LLBtorque4_param_12,
	.param .u64 LLBtorque4_param_13,
	.param .f32 LLBtorque4_param_14,
	.param .u64 LLBtorque4_param_15,
	.param .u64 LLBtorque4_param_16,
	.param .u64 LLBtorque4_param_17,
	.param .u64 LLBtorque4_param_18,
	.param .u64 LLBtorque4_param_19,
	.param .u64 LLBtorque4_param_20,
	.param .u64 LLBtorque4_param_21,
	.param .f32 LLBtorque4_param_22,
	.param .u64 LLBtorque4_param_23,
	.param .f32 LLBtorque4_param_24,
	.param .u32 LLBtorque4_param_25
)
{
	.reg .pred 	%p<81>;
	.reg .f32 	%f<536>;
	.reg .b32 	%r<65>;
	.reg .f64 	%fd<10>;
	.reg .b64 	%rd<69>;


	ld.param.u64 	%rd5, [LLBtorque4_param_3];
	ld.param.u64 	%rd6, [LLBtorque4_param_4];
	ld.param.u64 	%rd7, [LLBtorque4_param_5];
	ld.param.u64 	%rd8, [LLBtorque4_param_6];
	ld.param.u64 	%rd9, [LLBtorque4_param_7];
	ld.param.u64 	%rd10, [LLBtorque4_param_8];
	ld.param.u64 	%rd11, [LLBtorque4_param_9];
	ld.param.f32 	%f516, [LLBtorque4_param_10];
	ld.param.u64 	%rd12, [LLBtorque4_param_11];
	ld.param.f32 	%f517, [LLBtorque4_param_12];
	ld.param.u64 	%rd13, [LLBtorque4_param_15];
	ld.param.u64 	%rd14, [LLBtorque4_param_16];
	ld.param.u64 	%rd15, [LLBtorque4_param_17];
	ld.param.u64 	%rd16, [LLBtorque4_param_18];
	ld.param.u64 	%rd17, [LLBtorque4_param_19];
	ld.param.u64 	%rd18, [LLBtorque4_param_20];
	ld.param.u64 	%rd19, [LLBtorque4_param_21];
	ld.param.f32 	%f519, [LLBtorque4_param_22];
	ld.param.u64 	%rd20, [LLBtorque4_param_23];
	ld.param.f32 	%f518, [LLBtorque4_param_24];
	ld.param.u32 	%r2, [LLBtorque4_param_25];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p4, %r1, %r2;
	@%p4 bra 	BB0_55;

	cvta.to.global.u64 	%rd21, %rd5;
	mul.wide.s32 	%rd22, %r1, 4;
	add.s64 	%rd23, %rd21, %rd22;
	ld.global.f32 	%f1, [%rd23];
	cvta.to.global.u64 	%rd24, %rd6;
	add.s64 	%rd25, %rd24, %rd22;
	ld.global.f32 	%f2, [%rd25];
	cvta.to.global.u64 	%rd26, %rd7;
	add.s64 	%rd27, %rd26, %rd22;
	ld.global.f32 	%f3, [%rd27];
	cvta.to.global.u64 	%rd28, %rd8;
	add.s64 	%rd29, %rd28, %rd22;
	ld.global.f32 	%f4, [%rd29];
	cvta.to.global.u64 	%rd30, %rd9;
	add.s64 	%rd31, %rd30, %rd22;
	ld.global.f32 	%f5, [%rd31];
	cvta.to.global.u64 	%rd32, %rd10;
	add.s64 	%rd33, %rd32, %rd22;
	ld.global.f32 	%f6, [%rd33];
	setp.eq.s64	%p5, %rd11, 0;
	@%p5 bra 	BB0_3;

	cvta.to.global.u64 	%rd34, %rd11;
	add.s64 	%rd36, %rd34, %rd22;
	ld.global.f32 	%f97, [%rd36];
	mul.f32 	%f516, %f97, %f516;

BB0_3:
	setp.eq.s64	%p6, %rd12, 0;
	@%p6 bra 	BB0_5;

	cvta.to.global.u64 	%rd37, %rd12;
	add.s64 	%rd39, %rd37, %rd22;
	ld.global.f32 	%f98, [%rd39];
	mul.f32 	%f517, %f98, %f517;

BB0_5:
	setp.eq.s64	%p7, %rd20, 0;
	@%p7 bra 	BB0_7;

	cvta.to.global.u64 	%rd40, %rd20;
	add.s64 	%rd42, %rd40, %rd22;
	ld.global.f32 	%f99, [%rd42];
	mul.f32 	%f518, %f99, %f518;

BB0_7:
	setp.eq.s64	%p8, %rd19, 0;
	@%p8 bra 	BB0_9;

	cvta.to.global.u64 	%rd43, %rd19;
	add.s64 	%rd45, %rd43, %rd22;
	ld.global.f32 	%f100, [%rd45];
	mul.f32 	%f519, %f100, %f519;

BB0_9:
	setp.eq.f32	%p9, %f519, 0f00000000;
	selp.f32	%f101, 0f38D1B717, %f519, %p9;
	cvt.f64.f32	%fd2, %f101;
	cvt.f64.f32	%fd1, %f517;
	add.f64 	%fd3, %fd1, %fd1;
	setp.gt.f64	%p10, %fd2, %fd3;
	add.f32 	%f102, %f517, %f517;
	selp.f32	%f520, %f102, %f101, %p10;
	setp.neu.f32	%p11, %f520, %f517;
	@%p11 bra 	BB0_11;

	add.f64 	%fd4, %fd1, 0dBF847AE147AE147B;
	cvt.rn.f32.f64	%f520, %fd4;

BB0_11:
	cvta.to.global.u64 	%rd46, %rd18;
	cvta.to.global.u64 	%rd47, %rd17;
	cvta.to.global.u64 	%rd48, %rd16;
	cvta.to.global.u64 	%rd49, %rd15;
	cvta.to.global.u64 	%rd50, %rd14;
	cvta.to.global.u64 	%rd51, %rd13;
	add.s64 	%rd53, %rd51, %rd22;
	ld.global.f32 	%f18, [%rd53];
	add.s64 	%rd54, %rd50, %rd22;
	ld.global.f32 	%f19, [%rd54];
	add.s64 	%rd55, %rd49, %rd22;
	ld.global.f32 	%f20, [%rd55];
	add.s64 	%rd56, %rd48, %rd22;
	ld.global.f32 	%f21, [%rd56];
	add.s64 	%rd57, %rd47, %rd22;
	ld.global.f32 	%f22, [%rd57];
	add.s64 	%rd1, %rd46, %rd22;
	mul.f32 	%f103, %f2, %f2;
	fma.rn.f32 	%f104, %f1, %f1, %f103;
	fma.rn.f32 	%f23, %f3, %f3, %f104;
	mul.f32 	%f533, %f1, 0f00000000;
	mul.f32 	%f534, %f2, 0f00000000;
	mul.f32 	%f535, %f3, 0f00000000;
	setp.eq.f32	%p12, %f23, 0f00000000;
	setp.eq.f32	%p13, %f517, 0f00000000;
	or.pred  	%p14, %p12, %p13;
	@%p14 bra 	BB0_54;

	ld.global.f32 	%f27, [%rd1];
	add.f32 	%f105, %f516, %f516;
	mul.f32 	%f106, %f105, %f520;
	mul.f32 	%f107, %f517, 0f40400000;
	div.rn.f32 	%f28, %f106, %f107;
	setp.gtu.f32	%p15, %f520, %f517;
	@%p15 bra 	BB0_52;
	bra.uni 	BB0_13;

BB0_52:
	sub.f32 	%f401, %f520, %f517;
	mov.f32 	%f402, 0f347DE56D;
	div.rn.f32 	%f403, %f402, %f401;
	cvt.f64.f32	%fd8, %f403;
	setp.gt.f64	%p79, %fd8, 0d3EB0C6F7A0B5ED8D;
	selp.f32	%f404, 0f358637BD, %f403, %p79;
	div.rn.f32 	%f405, %f517, %f401;
	mul.f32 	%f406, %f23, 0f3F19999A;
	fma.rn.f32 	%f407, %f406, %f405, 0f3F800000;
	cvt.f64.f32	%fd9, %f404;
	mov.f32 	%f408, 0fBF800000;
	div.rn.f32 	%f409, %f408, %f404;
	setp.lt.f64	%p80, %fd9, 0d3DD5FD7FE1796495;
	selp.f32	%f410, 0fD03A43B7, %f409, %p80;
	mul.f32 	%f531, %f407, %f410;
	mov.f32 	%f532, %f28;
	bra.uni 	BB0_53;

BB0_13:
	setp.eq.f32	%p16, %f518, 0f00000000;
	selp.f32	%f29, 0f3F800000, %f518, %p16;
	mul.f32 	%f110, %f29, 0f3F000000;
	cvt.rzi.f32.f32	%f111, %f110;
	fma.rn.f32 	%f112, %f111, 0fC0000000, %f29;
	abs.f32 	%f30, %f112;
	div.rn.f32 	%f31, %f520, %f517;
	abs.f32 	%f32, %f31;
	setp.lt.f32	%p17, %f32, 0f00800000;
	mul.f32 	%f113, %f32, 0f4B800000;
	selp.f32	%f114, 0fC3170000, 0fC2FE0000, %p17;
	selp.f32	%f115, %f113, %f32, %p17;
	mov.b32 	 %r9, %f115;
	and.b32  	%r10, %r9, 8388607;
	or.b32  	%r11, %r10, 1065353216;
	mov.b32 	 %f116, %r11;
	shr.u32 	%r12, %r9, 23;
	cvt.rn.f32.u32	%f117, %r12;
	add.f32 	%f118, %f114, %f117;
	setp.gt.f32	%p18, %f116, 0f3FB504F3;
	mul.f32 	%f119, %f116, 0f3F000000;
	add.f32 	%f120, %f118, 0f3F800000;
	selp.f32	%f121, %f119, %f116, %p18;
	selp.f32	%f122, %f120, %f118, %p18;
	add.f32 	%f123, %f121, 0fBF800000;
	add.f32 	%f109, %f121, 0f3F800000;
	// inline asm
	rcp.approx.ftz.f32 %f108,%f109;
	// inline asm
	add.f32 	%f124, %f123, %f123;
	mul.f32 	%f125, %f108, %f124;
	mul.f32 	%f126, %f125, %f125;
	mov.f32 	%f127, 0f3C4CAF63;
	mov.f32 	%f128, 0f3B18F0FE;
	fma.rn.f32 	%f129, %f128, %f126, %f127;
	mov.f32 	%f130, 0f3DAAAABD;
	fma.rn.f32 	%f131, %f129, %f126, %f130;
	mul.rn.f32 	%f132, %f131, %f126;
	mul.rn.f32 	%f133, %f132, %f125;
	sub.f32 	%f134, %f123, %f125;
	neg.f32 	%f135, %f125;
	add.f32 	%f136, %f134, %f134;
	fma.rn.f32 	%f137, %f135, %f123, %f136;
	mul.rn.f32 	%f138, %f108, %f137;
	add.f32 	%f139, %f133, %f125;
	sub.f32 	%f140, %f125, %f139;
	add.f32 	%f141, %f133, %f140;
	add.f32 	%f142, %f138, %f141;
	add.f32 	%f143, %f139, %f142;
	sub.f32 	%f144, %f139, %f143;
	add.f32 	%f145, %f142, %f144;
	mov.f32 	%f146, 0f3F317200;
	mul.rn.f32 	%f147, %f122, %f146;
	mov.f32 	%f148, 0f35BFBE8E;
	mul.rn.f32 	%f149, %f122, %f148;
	add.f32 	%f150, %f147, %f143;
	sub.f32 	%f151, %f147, %f150;
	add.f32 	%f152, %f143, %f151;
	add.f32 	%f153, %f145, %f152;
	add.f32 	%f154, %f149, %f153;
	add.f32 	%f155, %f150, %f154;
	sub.f32 	%f156, %f150, %f155;
	add.f32 	%f157, %f154, %f156;
	abs.f32 	%f33, %f29;
	setp.gt.f32	%p19, %f33, 0f77F684DF;
	mul.f32 	%f158, %f29, 0f39000000;
	selp.f32	%f159, %f158, %f29, %p19;
	mul.rn.f32 	%f160, %f159, %f155;
	neg.f32 	%f161, %f160;
	fma.rn.f32 	%f162, %f159, %f155, %f161;
	fma.rn.f32 	%f163, %f159, %f157, %f162;
	mov.f32 	%f164, 0f00000000;
	fma.rn.f32 	%f165, %f164, %f155, %f163;
	add.rn.f32 	%f166, %f160, %f165;
	neg.f32 	%f167, %f166;
	add.rn.f32 	%f168, %f160, %f167;
	add.rn.f32 	%f169, %f168, %f165;
	mov.b32 	 %r13, %f166;
	setp.eq.s32	%p20, %r13, 1118925336;
	add.s32 	%r14, %r13, -1;
	mov.b32 	 %f170, %r14;
	add.f32 	%f171, %f169, 0f37000000;
	selp.f32	%f172, %f170, %f166, %p20;
	selp.f32	%f34, %f171, %f169, %p20;
	mul.f32 	%f173, %f172, 0f3FB8AA3B;
	cvt.rzi.f32.f32	%f174, %f173;
	mov.f32 	%f175, 0fBF317200;
	fma.rn.f32 	%f176, %f174, %f175, %f172;
	mov.f32 	%f177, 0fB5BFBE8E;
	fma.rn.f32 	%f178, %f174, %f177, %f176;
	mul.f32 	%f179, %f178, 0f3FB8AA3B;
	ex2.approx.ftz.f32 	%f180, %f179;
	add.f32 	%f181, %f174, 0f00000000;
	ex2.approx.f32 	%f182, %f181;
	mul.f32 	%f183, %f180, %f182;
	setp.lt.f32	%p21, %f172, 0fC2D20000;
	selp.f32	%f184, 0f00000000, %f183, %p21;
	setp.gt.f32	%p22, %f172, 0f42D20000;
	selp.f32	%f521, 0f7F800000, %f184, %p22;
	setp.eq.f32	%p23, %f521, 0f7F800000;
	@%p23 bra 	BB0_15;

	fma.rn.f32 	%f521, %f521, %f34, %f521;

BB0_15:
	setp.lt.f32	%p24, %f31, 0f00000000;
	setp.eq.f32	%p25, %f30, 0f3F800000;
	and.pred  	%p1, %p24, %p25;
	mov.b32 	 %r15, %f521;
	xor.b32  	%r16, %r15, -2147483648;
	mov.b32 	 %f185, %r16;
	selp.f32	%f523, %f185, %f521, %p1;
	setp.eq.f32	%p26, %f31, 0f00000000;
	@%p26 bra 	BB0_18;
	bra.uni 	BB0_16;

BB0_18:
	add.f32 	%f187, %f31, %f31;
	mov.b32 	 %r17, %f187;
	selp.b32	%r18, %r17, 0, %p25;
	or.b32  	%r19, %r18, 2139095040;
	setp.lt.f32	%p30, %f29, 0f00000000;
	selp.b32	%r20, %r19, %r18, %p30;
	mov.b32 	 %f523, %r20;
	bra.uni 	BB0_19;

BB0_16:
	setp.geu.f32	%p27, %f31, 0f00000000;
	@%p27 bra 	BB0_19;

	cvt.rzi.f32.f32	%f186, %f29;
	setp.neu.f32	%p28, %f186, %f29;
	selp.f32	%f523, 0f7FFFFFFF, %f523, %p28;

BB0_19:
	abs.f32 	%f510, %f31;
	abs.f32 	%f477, %f29;
	add.f32 	%f188, %f510, %f477;
	mov.b32 	 %r21, %f188;
	setp.lt.s32	%p31, %r21, 2139095040;
	@%p31 bra 	BB0_26;

	abs.f32 	%f513, %f31;
	abs.f32 	%f490, %f29;
	setp.gtu.f32	%p32, %f513, 0f7F800000;
	setp.gtu.f32	%p33, %f490, 0f7F800000;
	or.pred  	%p34, %p32, %p33;
	@%p34 bra 	BB0_25;
	bra.uni 	BB0_21;

BB0_25:
	add.f32 	%f523, %f29, %f31;
	bra.uni 	BB0_26;

BB0_21:
	abs.f32 	%f491, %f29;
	setp.eq.f32	%p35, %f491, 0f7F800000;
	@%p35 bra 	BB0_24;
	bra.uni 	BB0_22;

BB0_24:
	abs.f32 	%f515, %f31;
	setp.gt.f32	%p38, %f515, 0f3F800000;
	selp.b32	%r25, 2139095040, 0, %p38;
	xor.b32  	%r26, %r25, 2139095040;
	setp.lt.f32	%p39, %f29, 0f00000000;
	selp.b32	%r27, %r26, %r25, %p39;
	mov.b32 	 %f189, %r27;
	setp.eq.f32	%p40, %f31, 0fBF800000;
	selp.f32	%f523, 0f3F800000, %f189, %p40;
	bra.uni 	BB0_26;

BB0_22:
	abs.f32 	%f514, %f31;
	setp.neu.f32	%p36, %f514, 0f7F800000;
	@%p36 bra 	BB0_26;

	setp.ltu.f32	%p37, %f29, 0f00000000;
	selp.b32	%r22, 0, 2139095040, %p37;
	or.b32  	%r23, %r22, -2147483648;
	selp.b32	%r24, %r23, %r22, %p1;
	mov.b32 	 %f523, %r24;

BB0_26:
	mov.f32 	%f512, 0fB5BFBE8E;
	mov.f32 	%f511, 0fBF317200;
	mov.f32 	%f483, 0f00000000;
	mov.f32 	%f482, 0f35BFBE8E;
	mov.f32 	%f481, 0f3F317200;
	mov.f32 	%f480, 0f3DAAAABD;
	mov.f32 	%f479, 0f3C4CAF63;
	mov.f32 	%f478, 0f3B18F0FE;
	setp.eq.f32	%p41, %f31, 0f3F800000;
	selp.f32	%f46, 0f3F800000, %f523, %p41;
	abs.f32 	%f48, %f46;
	setp.lt.f32	%p42, %f48, 0f00800000;
	mul.f32 	%f195, %f48, 0f4B800000;
	selp.f32	%f196, 0fC3170000, 0fC2FE0000, %p42;
	selp.f32	%f197, %f195, %f48, %p42;
	mov.b32 	 %r28, %f197;
	and.b32  	%r29, %r28, 8388607;
	or.b32  	%r30, %r29, 1065353216;
	mov.b32 	 %f198, %r30;
	shr.u32 	%r31, %r28, 23;
	cvt.rn.f32.u32	%f199, %r31;
	add.f32 	%f200, %f196, %f199;
	setp.gt.f32	%p43, %f198, 0f3FB504F3;
	mul.f32 	%f201, %f198, 0f3F000000;
	add.f32 	%f202, %f200, 0f3F800000;
	selp.f32	%f203, %f201, %f198, %p43;
	selp.f32	%f204, %f202, %f200, %p43;
	add.f32 	%f205, %f203, 0fBF800000;
	add.f32 	%f191, %f203, 0f3F800000;
	// inline asm
	rcp.approx.ftz.f32 %f190,%f191;
	// inline asm
	add.f32 	%f206, %f205, %f205;
	mul.f32 	%f207, %f190, %f206;
	mul.f32 	%f208, %f207, %f207;
	fma.rn.f32 	%f211, %f478, %f208, %f479;
	fma.rn.f32 	%f213, %f211, %f208, %f480;
	mul.rn.f32 	%f214, %f213, %f208;
	mul.rn.f32 	%f215, %f214, %f207;
	sub.f32 	%f216, %f205, %f207;
	neg.f32 	%f217, %f207;
	add.f32 	%f218, %f216, %f216;
	fma.rn.f32 	%f219, %f217, %f205, %f218;
	mul.rn.f32 	%f220, %f190, %f219;
	add.f32 	%f221, %f215, %f207;
	sub.f32 	%f222, %f207, %f221;
	add.f32 	%f223, %f215, %f222;
	add.f32 	%f224, %f220, %f223;
	add.f32 	%f225, %f221, %f224;
	sub.f32 	%f226, %f221, %f225;
	add.f32 	%f227, %f224, %f226;
	mul.rn.f32 	%f229, %f204, %f481;
	mul.rn.f32 	%f231, %f204, %f482;
	add.f32 	%f232, %f229, %f225;
	sub.f32 	%f233, %f229, %f232;
	add.f32 	%f234, %f225, %f233;
	add.f32 	%f235, %f227, %f234;
	add.f32 	%f236, %f231, %f235;
	add.f32 	%f237, %f232, %f236;
	sub.f32 	%f238, %f232, %f237;
	add.f32 	%f239, %f236, %f238;
	mov.f32 	%f240, 0f3F9DDC72;
	mul.rn.f32 	%f241, %f240, %f237;
	neg.f32 	%f242, %f241;
	fma.rn.f32 	%f243, %f240, %f237, %f242;
	fma.rn.f32 	%f244, %f240, %f239, %f243;
	fma.rn.f32 	%f246, %f483, %f237, %f244;
	add.rn.f32 	%f247, %f241, %f246;
	neg.f32 	%f248, %f247;
	add.rn.f32 	%f249, %f241, %f248;
	add.rn.f32 	%f250, %f249, %f246;
	mov.b32 	 %r32, %f247;
	setp.eq.s32	%p44, %r32, 1118925336;
	add.s32 	%r33, %r32, -1;
	mov.b32 	 %f251, %r33;
	add.f32 	%f252, %f250, 0f37000000;
	selp.f32	%f253, %f251, %f247, %p44;
	selp.f32	%f49, %f252, %f250, %p44;
	mul.f32 	%f254, %f253, 0f3FB8AA3B;
	cvt.rzi.f32.f32	%f255, %f254;
	fma.rn.f32 	%f257, %f255, %f511, %f253;
	fma.rn.f32 	%f259, %f255, %f512, %f257;
	mul.f32 	%f260, %f259, 0f3FB8AA3B;
	ex2.approx.ftz.f32 	%f261, %f260;
	add.f32 	%f262, %f255, 0f00000000;
	ex2.approx.f32 	%f263, %f262;
	mul.f32 	%f264, %f261, %f263;
	setp.lt.f32	%p45, %f253, 0fC2D20000;
	selp.f32	%f265, 0f00000000, %f264, %p45;
	setp.gt.f32	%p46, %f253, 0f42D20000;
	selp.f32	%f524, 0f7F800000, %f265, %p46;
	setp.eq.f32	%p47, %f524, 0f7F800000;
	@%p47 bra 	BB0_28;

	fma.rn.f32 	%f524, %f524, %f49, %f524;

BB0_28:
	mov.f32 	%f495, 0f3F1DDC72;
	cvt.rzi.f32.f32	%f494, %f495;
	fma.rn.f32 	%f493, %f494, 0fC0000000, 0f3F9DDC72;
	abs.f32 	%f492, %f493;
	setp.lt.f32	%p48, %f46, 0f00000000;
	setp.eq.f32	%p49, %f492, 0f3F800000;
	and.pred  	%p2, %p48, %p49;
	mov.b32 	 %r34, %f524;
	xor.b32  	%r35, %r34, -2147483648;
	mov.b32 	 %f266, %r35;
	selp.f32	%f526, %f266, %f524, %p2;
	setp.eq.f32	%p50, %f46, 0f00000000;
	@%p50 bra 	BB0_31;
	bra.uni 	BB0_29;

BB0_31:
	add.f32 	%f269, %f46, %f46;
	selp.f32	%f526, %f269, 0f00000000, %p49;
	bra.uni 	BB0_32;

BB0_29:
	setp.geu.f32	%p51, %f46, 0f00000000;
	@%p51 bra 	BB0_32;

	mov.f32 	%f503, 0f3F9DDC72;
	cvt.rzi.f32.f32	%f268, %f503;
	setp.neu.f32	%p52, %f268, 0f3F9DDC72;
	selp.f32	%f526, 0f7FFFFFFF, %f526, %p52;

BB0_32:
	abs.f32 	%f496, %f46;
	add.f32 	%f270, %f496, 0f3F9DDC72;
	mov.b32 	 %r36, %f270;
	setp.lt.s32	%p54, %r36, 2139095040;
	@%p54 bra 	BB0_37;

	abs.f32 	%f501, %f46;
	setp.gtu.f32	%p55, %f501, 0f7F800000;
	@%p55 bra 	BB0_36;
	bra.uni 	BB0_34;

BB0_36:
	add.f32 	%f526, %f46, 0f3F9DDC72;
	bra.uni 	BB0_37;

BB0_34:
	abs.f32 	%f502, %f46;
	setp.neu.f32	%p56, %f502, 0f7F800000;
	@%p56 bra 	BB0_37;

	selp.f32	%f526, 0fFF800000, 0f7F800000, %p2;

BB0_37:
	mov.f32 	%f498, 0fB5BFBE8E;
	mov.f32 	%f497, 0fBF317200;
	mov.f32 	%f489, 0f00000000;
	mov.f32 	%f488, 0f35BFBE8E;
	mov.f32 	%f487, 0f3F317200;
	mov.f32 	%f486, 0f3DAAAABD;
	mov.f32 	%f485, 0f3C4CAF63;
	mov.f32 	%f484, 0f3B18F0FE;
	mov.f32 	%f273, 0f3F800000;
	sub.f32 	%f274, %f273, %f526;
	setp.eq.f32	%p57, %f46, 0f3F800000;
	selp.f32	%f60, 0f00000000, %f274, %p57;
	abs.f32 	%f62, %f60;
	setp.lt.f32	%p58, %f62, 0f00800000;
	mul.f32 	%f278, %f62, 0f4B800000;
	selp.f32	%f279, 0fC3170000, 0fC2FE0000, %p58;
	selp.f32	%f280, %f278, %f62, %p58;
	mov.b32 	 %r37, %f280;
	and.b32  	%r38, %r37, 8388607;
	or.b32  	%r39, %r38, 1065353216;
	mov.b32 	 %f281, %r39;
	shr.u32 	%r40, %r37, 23;
	cvt.rn.f32.u32	%f282, %r40;
	add.f32 	%f283, %f279, %f282;
	setp.gt.f32	%p59, %f281, 0f3FB504F3;
	mul.f32 	%f284, %f281, 0f3F000000;
	add.f32 	%f285, %f283, 0f3F800000;
	selp.f32	%f286, %f284, %f281, %p59;
	selp.f32	%f287, %f285, %f283, %p59;
	add.f32 	%f288, %f286, 0fBF800000;
	add.f32 	%f272, %f286, 0f3F800000;
	// inline asm
	rcp.approx.ftz.f32 %f271,%f272;
	// inline asm
	add.f32 	%f289, %f288, %f288;
	mul.f32 	%f290, %f271, %f289;
	mul.f32 	%f291, %f290, %f290;
	fma.rn.f32 	%f294, %f484, %f291, %f485;
	fma.rn.f32 	%f296, %f294, %f291, %f486;
	mul.rn.f32 	%f297, %f296, %f291;
	mul.rn.f32 	%f298, %f297, %f290;
	sub.f32 	%f299, %f288, %f290;
	neg.f32 	%f300, %f290;
	add.f32 	%f301, %f299, %f299;
	fma.rn.f32 	%f302, %f300, %f288, %f301;
	mul.rn.f32 	%f303, %f271, %f302;
	add.f32 	%f304, %f298, %f290;
	sub.f32 	%f305, %f290, %f304;
	add.f32 	%f306, %f298, %f305;
	add.f32 	%f307, %f303, %f306;
	add.f32 	%f308, %f304, %f307;
	sub.f32 	%f309, %f304, %f308;
	add.f32 	%f310, %f307, %f309;
	mul.rn.f32 	%f312, %f287, %f487;
	mul.rn.f32 	%f314, %f287, %f488;
	add.f32 	%f315, %f312, %f308;
	sub.f32 	%f316, %f312, %f315;
	add.f32 	%f317, %f308, %f316;
	add.f32 	%f318, %f310, %f317;
	add.f32 	%f319, %f314, %f318;
	add.f32 	%f320, %f315, %f319;
	sub.f32 	%f321, %f315, %f320;
	add.f32 	%f322, %f319, %f321;
	mov.f32 	%f323, 0f3EDE2AC3;
	mul.rn.f32 	%f324, %f323, %f320;
	neg.f32 	%f325, %f324;
	fma.rn.f32 	%f326, %f323, %f320, %f325;
	fma.rn.f32 	%f327, %f323, %f322, %f326;
	fma.rn.f32 	%f329, %f489, %f320, %f327;
	add.rn.f32 	%f330, %f324, %f329;
	neg.f32 	%f331, %f330;
	add.rn.f32 	%f332, %f324, %f331;
	add.rn.f32 	%f333, %f332, %f329;
	mov.b32 	 %r41, %f330;
	setp.eq.s32	%p60, %r41, 1118925336;
	add.s32 	%r42, %r41, -1;
	mov.b32 	 %f334, %r42;
	add.f32 	%f335, %f333, 0f37000000;
	selp.f32	%f336, %f334, %f330, %p60;
	selp.f32	%f63, %f335, %f333, %p60;
	mul.f32 	%f337, %f336, 0f3FB8AA3B;
	cvt.rzi.f32.f32	%f338, %f337;
	fma.rn.f32 	%f340, %f338, %f497, %f336;
	fma.rn.f32 	%f342, %f338, %f498, %f340;
	mul.f32 	%f343, %f342, 0f3FB8AA3B;
	ex2.approx.ftz.f32 	%f344, %f343;
	add.f32 	%f345, %f338, 0f00000000;
	ex2.approx.f32 	%f346, %f345;
	mul.f32 	%f347, %f344, %f346;
	setp.lt.f32	%p61, %f336, 0fC2D20000;
	selp.f32	%f348, 0f00000000, %f347, %p61;
	setp.gt.f32	%p62, %f336, 0f42D20000;
	selp.f32	%f527, 0f7F800000, %f348, %p62;
	setp.eq.f32	%p63, %f527, 0f7F800000;
	@%p63 bra 	BB0_39;

	fma.rn.f32 	%f527, %f527, %f63, %f527;

BB0_39:
	mov.f32 	%f507, 0f3E5E2AC3;
	cvt.rzi.f32.f32	%f506, %f507;
	fma.rn.f32 	%f505, %f506, 0fC0000000, 0f3EDE2AC3;
	abs.f32 	%f504, %f505;
	setp.lt.f32	%p64, %f60, 0f00000000;
	setp.eq.f32	%p65, %f504, 0f3F800000;
	and.pred  	%p3, %p64, %p65;
	mov.b32 	 %r43, %f527;
	xor.b32  	%r44, %r43, -2147483648;
	mov.b32 	 %f349, %r44;
	selp.f32	%f529, %f349, %f527, %p3;
	setp.eq.f32	%p66, %f60, 0f00000000;
	@%p66 bra 	BB0_42;
	bra.uni 	BB0_40;

BB0_42:
	add.f32 	%f352, %f60, %f60;
	selp.f32	%f529, %f352, 0f00000000, %p65;
	bra.uni 	BB0_43;

BB0_40:
	setp.geu.f32	%p67, %f60, 0f00000000;
	@%p67 bra 	BB0_43;

	mov.f32 	%f509, 0f3EDE2AC3;
	cvt.rzi.f32.f32	%f351, %f509;
	setp.neu.f32	%p68, %f351, 0f3EDE2AC3;
	selp.f32	%f529, 0f7FFFFFFF, %f529, %p68;

BB0_43:
	add.f32 	%f353, %f62, 0f3EDE2AC3;
	mov.b32 	 %r45, %f353;
	setp.lt.s32	%p70, %r45, 2139095040;
	@%p70 bra 	BB0_48;

	setp.gtu.f32	%p71, %f62, 0f7F800000;
	@%p71 bra 	BB0_47;
	bra.uni 	BB0_45;

BB0_47:
	add.f32 	%f529, %f60, 0f3EDE2AC3;
	bra.uni 	BB0_48;

BB0_45:
	setp.neu.f32	%p72, %f62, 0f7F800000;
	@%p72 bra 	BB0_48;

	selp.f32	%f529, 0fFF800000, 0f7F800000, %p3;

BB0_48:
	setp.eq.f32	%p73, %f60, 0f3F800000;
	selp.f32	%f354, 0f3F800000, %f529, %p73;
	cvt.f64.f32	%fd5, %f354;
	setp.lt.f64	%p74, %fd5, 0d3F50624DD2F1A9FC;
	selp.f32	%f74, 0f3A83126F, %f354, %p74;
	mov.f32 	%f355, 0f40400000;
	div.rn.f32 	%f75, %f355, %f46;
	mul.f32 	%f76, %f75, %f74;
	mul.f32 	%f77, %f76, %f76;
	abs.f32 	%f78, %f76;
	setp.ltu.f32	%p75, %f78, 0f3F800000;
	@%p75 bra 	BB0_50;
	bra.uni 	BB0_49;

BB0_50:
	mov.f32 	%f372, 0f394FFF49;
	mov.f32 	%f373, 0f363D0ADA;
	fma.rn.f32 	%f374, %f373, %f77, %f372;
	mov.f32 	%f375, 0f3C08889A;
	fma.rn.f32 	%f376, %f374, %f77, %f375;
	mov.f32 	%f377, 0f3E2AAAAB;
	fma.rn.f32 	%f378, %f376, %f77, %f377;
	mul.f32 	%f379, %f77, %f378;
	fma.rn.f32 	%f530, %f379, %f76, %f76;
	bra.uni 	BB0_51;

BB0_49:
	mov.f32 	%f500, 0fB5BFBE8E;
	mov.f32 	%f499, 0fBF317200;
	mul.f32 	%f356, %f78, 0f3FB8AA3B;
	cvt.rzi.f32.f32	%f357, %f356;
	fma.rn.f32 	%f359, %f357, %f499, %f78;
	fma.rn.f32 	%f361, %f357, %f500, %f359;
	mul.f32 	%f362, %f361, 0f3FB8AA3B;
	ex2.approx.ftz.f32 	%f363, %f362;
	add.f32 	%f364, %f357, 0fC0000000;
	ex2.approx.f32 	%f365, %f364;
	mul.f32 	%f366, %f363, %f365;
	mov.f32 	%f367, 0f3E000000;
	div.approx.f32 	%f368, %f367, %f366;
	neg.f32 	%f369, %f368;
	mov.f32 	%f370, 0f40000000;
	fma.rn.f32 	%f371, %f370, %f366, %f369;
	mov.b32 	 %r46, %f371;
	setp.ltu.f32	%p76, %f78, 0f42B40000;
	selp.b32	%r47, %r46, 2139095040, %p76;
	mov.b32 	 %r48, %f76;
	and.b32  	%r49, %r48, -2147483648;
	or.b32  	%r50, %r47, %r49;
	mov.b32 	 %f530, %r50;

BB0_51:
	mov.f32 	%f508, 0f3F800000;
	mul.f32 	%f380, %f530, %f530;
	rcp.rn.f32 	%f381, %f380;
	rcp.rn.f32 	%f382, %f77;
	sub.f32 	%f383, %f382, %f381;
	mul.f32 	%f384, %f75, %f383;
	sub.f32 	%f386, %f508, %f384;
	div.rn.f32 	%f387, %f383, %f386;
	mul.f32 	%f388, %f520, 0f19857725;
	rcp.rn.f32 	%f389, %f388;
	mul.f32 	%f390, %f389, 0f0FA575F3;
	mul.f32 	%f391, %f390, %f387;
	cvt.f64.f32	%fd6, %f391;
	setp.gt.f64	%p77, %fd6, 0d3EB0C6F7A0B5ED8D;
	selp.f32	%f392, 0f358637BD, %f391, %p77;
	div.rn.f32 	%f393, %f46, 0fC0400000;
	add.f32 	%f394, %f393, 0f3F800000;
	mul.f32 	%f532, %f516, %f394;
	add.f32 	%f395, %f392, %f392;
	mul.f32 	%f396, %f74, %f74;
	div.rn.f32 	%f397, %f23, %f396;
	sub.f32 	%f398, %f508, %f397;
	cvt.f64.f32	%fd7, %f392;
	rcp.rn.f32 	%f399, %f395;
	setp.lt.f64	%p78, %fd7, 0d3DD5FD7FE1796495;
	selp.f32	%f400, 0f4FBA43B7, %f399, %p78;
	mul.f32 	%f531, %f398, %f400;

BB0_53:
	mul.f32 	%f411, %f531, 0f35A8A9B8;
	mul.f32 	%f412, %f516, %f532;
	mul.f32 	%f413, %f532, %f412;
	sub.f32 	%f414, %f532, %f28;
	div.rn.f32 	%f415, %f414, %f413;
	sqrt.rn.f32 	%f416, %f415;
	div.rn.f32 	%f417, %f28, %f516;
	sqrt.rn.f32 	%f418, %f417;
	fma.rn.f32 	%f419, %f1, %f411, %f4;
	fma.rn.f32 	%f420, %f2, %f411, %f5;
	fma.rn.f32 	%f421, %f3, %f411, %f6;
	fma.rn.f32 	%f422, %f18, %f416, %f419;
	fma.rn.f32 	%f423, %f19, %f416, %f420;
	fma.rn.f32 	%f424, %f20, %f416, %f421;
	mul.f32 	%f425, %f2, %f421;
	mul.f32 	%f426, %f3, %f420;
	sub.f32 	%f427, %f425, %f426;
	mul.f32 	%f428, %f3, %f419;
	mul.f32 	%f429, %f1, %f421;
	sub.f32 	%f430, %f428, %f429;
	mul.f32 	%f431, %f1, %f420;
	mul.f32 	%f432, %f2, %f419;
	sub.f32 	%f433, %f431, %f432;
	mul.f32 	%f434, %f2, %f420;
	fma.rn.f32 	%f435, %f1, %f419, %f434;
	fma.rn.f32 	%f436, %f3, %f421, %f435;
	mul.f32 	%f437, %f2, %f424;
	mul.f32 	%f438, %f3, %f423;
	sub.f32 	%f439, %f437, %f438;
	mul.f32 	%f440, %f3, %f422;
	mul.f32 	%f441, %f1, %f424;
	sub.f32 	%f442, %f440, %f441;
	mul.f32 	%f443, %f1, %f423;
	mul.f32 	%f444, %f2, %f422;
	sub.f32 	%f445, %f443, %f444;
	mul.f32 	%f446, %f2, %f445;
	mul.f32 	%f447, %f3, %f442;
	sub.f32 	%f448, %f446, %f447;
	mul.f32 	%f449, %f3, %f439;
	mul.f32 	%f450, %f1, %f445;
	sub.f32 	%f451, %f449, %f450;
	mul.f32 	%f452, %f1, %f442;
	mul.f32 	%f453, %f2, %f439;
	sub.f32 	%f454, %f452, %f453;
	fma.rn.f32 	%f455, %f516, %f516, 0f3F800000;
	rcp.rn.f32 	%f456, %f455;
	mul.f32 	%f457, %f427, %f456;
	mul.f32 	%f458, %f430, %f456;
	mul.f32 	%f459, %f433, %f456;
	mul.f32 	%f460, %f28, %f456;
	div.rn.f32 	%f461, %f460, %f23;
	mul.f32 	%f462, %f436, %f461;
	mul.f32 	%f463, %f1, %f462;
	mul.f32 	%f464, %f2, %f462;
	mul.f32 	%f465, %f3, %f462;
	sub.f32 	%f466, %f463, %f457;
	sub.f32 	%f467, %f464, %f458;
	sub.f32 	%f468, %f465, %f459;
	mul.f32 	%f469, %f532, %f456;
	div.rn.f32 	%f470, %f469, %f23;
	mul.f32 	%f471, %f470, %f448;
	mul.f32 	%f472, %f470, %f451;
	mul.f32 	%f473, %f470, %f454;
	sub.f32 	%f474, %f466, %f471;
	sub.f32 	%f475, %f467, %f472;
	sub.f32 	%f476, %f468, %f473;
	fma.rn.f32 	%f533, %f21, %f418, %f474;
	fma.rn.f32 	%f534, %f22, %f418, %f475;
	fma.rn.f32 	%f535, %f27, %f418, %f476;

BB0_54:
	ld.param.u64 	%rd68, [LLBtorque4_param_2];
	ld.param.u64 	%rd67, [LLBtorque4_param_1];
	ld.param.u64 	%rd66, [LLBtorque4_param_0];
	mov.u32 	%r64, %ctaid.x;
	mov.u32 	%r63, %ctaid.y;
	mov.u32 	%r62, %nctaid.x;
	mov.u32 	%r61, %tid.x;
	mov.u32 	%r60, %ntid.x;
	mad.lo.s32 	%r59, %r62, %r63, %r64;
	mad.lo.s32 	%r58, %r59, %r60, %r61;
	mul.wide.s32 	%rd65, %r58, 4;
	cvta.to.global.u64 	%rd58, %rd66;
	add.s64 	%rd60, %rd58, %rd65;
	st.global.f32 	[%rd60], %f533;
	cvta.to.global.u64 	%rd61, %rd67;
	add.s64 	%rd62, %rd61, %rd65;
	st.global.f32 	[%rd62], %f534;
	cvta.to.global.u64 	%rd63, %rd68;
	add.s64 	%rd64, %rd63, %rd65;
	st.global.f32 	[%rd64], %f535;

BB0_55:
	ret;
}


`
)
