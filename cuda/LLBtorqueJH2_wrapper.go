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

// CUDA handle for LLBtorque2JH kernel
var LLBtorque2JH_code cu.Function

// Stores the arguments for LLBtorque2JH kernel invocation
type LLBtorque2JH_args_t struct {
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
	arg_tempJH     unsafe.Pointer
	arg_N          int
	argptr         [23]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for LLBtorque2JH kernel invocation
var LLBtorque2JH_args LLBtorque2JH_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	LLBtorque2JH_args.argptr[0] = unsafe.Pointer(&LLBtorque2JH_args.arg_tx)
	LLBtorque2JH_args.argptr[1] = unsafe.Pointer(&LLBtorque2JH_args.arg_ty)
	LLBtorque2JH_args.argptr[2] = unsafe.Pointer(&LLBtorque2JH_args.arg_tz)
	LLBtorque2JH_args.argptr[3] = unsafe.Pointer(&LLBtorque2JH_args.arg_mx)
	LLBtorque2JH_args.argptr[4] = unsafe.Pointer(&LLBtorque2JH_args.arg_my)
	LLBtorque2JH_args.argptr[5] = unsafe.Pointer(&LLBtorque2JH_args.arg_mz)
	LLBtorque2JH_args.argptr[6] = unsafe.Pointer(&LLBtorque2JH_args.arg_hx)
	LLBtorque2JH_args.argptr[7] = unsafe.Pointer(&LLBtorque2JH_args.arg_hy)
	LLBtorque2JH_args.argptr[8] = unsafe.Pointer(&LLBtorque2JH_args.arg_hz)
	LLBtorque2JH_args.argptr[9] = unsafe.Pointer(&LLBtorque2JH_args.arg_alpha_)
	LLBtorque2JH_args.argptr[10] = unsafe.Pointer(&LLBtorque2JH_args.arg_alpha_mul)
	LLBtorque2JH_args.argptr[11] = unsafe.Pointer(&LLBtorque2JH_args.arg_TCurie_)
	LLBtorque2JH_args.argptr[12] = unsafe.Pointer(&LLBtorque2JH_args.arg_TCurie_mul)
	LLBtorque2JH_args.argptr[13] = unsafe.Pointer(&LLBtorque2JH_args.arg_Msat_)
	LLBtorque2JH_args.argptr[14] = unsafe.Pointer(&LLBtorque2JH_args.arg_Msat_mul)
	LLBtorque2JH_args.argptr[15] = unsafe.Pointer(&LLBtorque2JH_args.arg_hth1x)
	LLBtorque2JH_args.argptr[16] = unsafe.Pointer(&LLBtorque2JH_args.arg_hth1y)
	LLBtorque2JH_args.argptr[17] = unsafe.Pointer(&LLBtorque2JH_args.arg_hth1z)
	LLBtorque2JH_args.argptr[18] = unsafe.Pointer(&LLBtorque2JH_args.arg_hth2x)
	LLBtorque2JH_args.argptr[19] = unsafe.Pointer(&LLBtorque2JH_args.arg_hth2y)
	LLBtorque2JH_args.argptr[20] = unsafe.Pointer(&LLBtorque2JH_args.arg_hth2z)
	LLBtorque2JH_args.argptr[21] = unsafe.Pointer(&LLBtorque2JH_args.arg_tempJH)
	LLBtorque2JH_args.argptr[22] = unsafe.Pointer(&LLBtorque2JH_args.arg_N)
}

// Wrapper for LLBtorque2JH CUDA kernel, asynchronous.
func k_LLBtorque2JH_async(tx unsafe.Pointer, ty unsafe.Pointer, tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, hx unsafe.Pointer, hy unsafe.Pointer, hz unsafe.Pointer, alpha_ unsafe.Pointer, alpha_mul float32, TCurie_ unsafe.Pointer, TCurie_mul float32, Msat_ unsafe.Pointer, Msat_mul float32, hth1x unsafe.Pointer, hth1y unsafe.Pointer, hth1z unsafe.Pointer, hth2x unsafe.Pointer, hth2y unsafe.Pointer, hth2z unsafe.Pointer, tempJH unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("LLBtorque2JH")
	}

	LLBtorque2JH_args.Lock()
	defer LLBtorque2JH_args.Unlock()

	if LLBtorque2JH_code == 0 {
		LLBtorque2JH_code = fatbinLoad(LLBtorque2JH_map, "LLBtorque2JH")
	}

	LLBtorque2JH_args.arg_tx = tx
	LLBtorque2JH_args.arg_ty = ty
	LLBtorque2JH_args.arg_tz = tz
	LLBtorque2JH_args.arg_mx = mx
	LLBtorque2JH_args.arg_my = my
	LLBtorque2JH_args.arg_mz = mz
	LLBtorque2JH_args.arg_hx = hx
	LLBtorque2JH_args.arg_hy = hy
	LLBtorque2JH_args.arg_hz = hz
	LLBtorque2JH_args.arg_alpha_ = alpha_
	LLBtorque2JH_args.arg_alpha_mul = alpha_mul
	LLBtorque2JH_args.arg_TCurie_ = TCurie_
	LLBtorque2JH_args.arg_TCurie_mul = TCurie_mul
	LLBtorque2JH_args.arg_Msat_ = Msat_
	LLBtorque2JH_args.arg_Msat_mul = Msat_mul
	LLBtorque2JH_args.arg_hth1x = hth1x
	LLBtorque2JH_args.arg_hth1y = hth1y
	LLBtorque2JH_args.arg_hth1z = hth1z
	LLBtorque2JH_args.arg_hth2x = hth2x
	LLBtorque2JH_args.arg_hth2y = hth2y
	LLBtorque2JH_args.arg_hth2z = hth2z
	LLBtorque2JH_args.arg_tempJH = tempJH
	LLBtorque2JH_args.arg_N = N

	args := LLBtorque2JH_args.argptr[:]
	cu.LaunchKernel(LLBtorque2JH_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("LLBtorque2JH")
	}
}

// maps compute capability on PTX code for LLBtorque2JH kernel.
var LLBtorque2JH_map = map[int]string{0: "",
	30: LLBtorque2JH_ptx_30}

// LLBtorque2JH PTX code for various compute capabilities.
const (
	LLBtorque2JH_ptx_30 = `
.version 6.5
.target sm_30
.address_size 64

	// .globl	LLBtorque2JH

.visible .entry LLBtorque2JH(
	.param .u64 LLBtorque2JH_param_0,
	.param .u64 LLBtorque2JH_param_1,
	.param .u64 LLBtorque2JH_param_2,
	.param .u64 LLBtorque2JH_param_3,
	.param .u64 LLBtorque2JH_param_4,
	.param .u64 LLBtorque2JH_param_5,
	.param .u64 LLBtorque2JH_param_6,
	.param .u64 LLBtorque2JH_param_7,
	.param .u64 LLBtorque2JH_param_8,
	.param .u64 LLBtorque2JH_param_9,
	.param .f32 LLBtorque2JH_param_10,
	.param .u64 LLBtorque2JH_param_11,
	.param .f32 LLBtorque2JH_param_12,
	.param .u64 LLBtorque2JH_param_13,
	.param .f32 LLBtorque2JH_param_14,
	.param .u64 LLBtorque2JH_param_15,
	.param .u64 LLBtorque2JH_param_16,
	.param .u64 LLBtorque2JH_param_17,
	.param .u64 LLBtorque2JH_param_18,
	.param .u64 LLBtorque2JH_param_19,
	.param .u64 LLBtorque2JH_param_20,
	.param .u64 LLBtorque2JH_param_21,
	.param .u32 LLBtorque2JH_param_22
)
{
	.reg .pred 	%p<49>;
	.reg .f32 	%f<396>;
	.reg .b32 	%r<41>;
	.reg .f64 	%fd<7>;
	.reg .b64 	%rd<65>;


	ld.param.u64 	%rd5, [LLBtorque2JH_param_3];
	ld.param.u64 	%rd6, [LLBtorque2JH_param_4];
	ld.param.u64 	%rd7, [LLBtorque2JH_param_5];
	ld.param.u64 	%rd8, [LLBtorque2JH_param_6];
	ld.param.u64 	%rd9, [LLBtorque2JH_param_7];
	ld.param.u64 	%rd10, [LLBtorque2JH_param_8];
	ld.param.u64 	%rd11, [LLBtorque2JH_param_9];
	ld.param.f32 	%f382, [LLBtorque2JH_param_10];
	ld.param.u64 	%rd12, [LLBtorque2JH_param_11];
	ld.param.f32 	%f383, [LLBtorque2JH_param_12];
	ld.param.u64 	%rd13, [LLBtorque2JH_param_15];
	ld.param.u64 	%rd14, [LLBtorque2JH_param_16];
	ld.param.u64 	%rd15, [LLBtorque2JH_param_17];
	ld.param.u64 	%rd16, [LLBtorque2JH_param_18];
	ld.param.u64 	%rd17, [LLBtorque2JH_param_19];
	ld.param.u64 	%rd18, [LLBtorque2JH_param_20];
	ld.param.u64 	%rd19, [LLBtorque2JH_param_21];
	ld.param.u32 	%r2, [LLBtorque2JH_param_22];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p3, %r1, %r2;
	@%p3 bra 	BB0_35;

	cvta.to.global.u64 	%rd20, %rd5;
	mul.wide.s32 	%rd21, %r1, 4;
	add.s64 	%rd22, %rd20, %rd21;
	ld.global.f32 	%f1, [%rd22];
	cvta.to.global.u64 	%rd23, %rd6;
	add.s64 	%rd24, %rd23, %rd21;
	ld.global.f32 	%f2, [%rd24];
	cvta.to.global.u64 	%rd25, %rd7;
	add.s64 	%rd26, %rd25, %rd21;
	ld.global.f32 	%f3, [%rd26];
	cvta.to.global.u64 	%rd27, %rd8;
	add.s64 	%rd28, %rd27, %rd21;
	ld.global.f32 	%f4, [%rd28];
	cvta.to.global.u64 	%rd29, %rd9;
	add.s64 	%rd30, %rd29, %rd21;
	ld.global.f32 	%f5, [%rd30];
	cvta.to.global.u64 	%rd31, %rd10;
	add.s64 	%rd32, %rd31, %rd21;
	ld.global.f32 	%f6, [%rd32];
	setp.eq.s64	%p4, %rd11, 0;
	@%p4 bra 	BB0_3;

	cvta.to.global.u64 	%rd33, %rd11;
	add.s64 	%rd35, %rd33, %rd21;
	ld.global.f32 	%f67, [%rd35];
	mul.f32 	%f382, %f67, %f382;

BB0_3:
	setp.eq.s64	%p5, %rd12, 0;
	@%p5 bra 	BB0_5;

	cvta.to.global.u64 	%rd36, %rd12;
	add.s64 	%rd38, %rd36, %rd21;
	ld.global.f32 	%f68, [%rd38];
	mul.f32 	%f383, %f68, %f383;

BB0_5:
	cvta.to.global.u64 	%rd39, %rd19;
	add.s64 	%rd41, %rd39, %rd21;
	ld.global.f32 	%f69, [%rd41];
	setp.eq.f32	%p6, %f69, 0f00000000;
	selp.f32	%f70, 0f38D1B717, %f69, %p6;
	cvt.f64.f32	%fd2, %f70;
	cvt.f64.f32	%fd1, %f383;
	add.f64 	%fd3, %fd1, %fd1;
	setp.gt.f64	%p7, %fd2, %fd3;
	add.f32 	%f71, %f383, %f383;
	selp.f32	%f384, %f71, %f70, %p7;
	setp.neu.f32	%p8, %f384, %f383;
	@%p8 bra 	BB0_7;

	add.f64 	%fd4, %fd1, 0dBF847AE147AE147B;
	cvt.rn.f32.f64	%f384, %fd4;

BB0_7:
	cvta.to.global.u64 	%rd42, %rd18;
	cvta.to.global.u64 	%rd43, %rd17;
	cvta.to.global.u64 	%rd44, %rd16;
	cvta.to.global.u64 	%rd45, %rd15;
	cvta.to.global.u64 	%rd46, %rd14;
	cvta.to.global.u64 	%rd47, %rd13;
	add.s64 	%rd49, %rd47, %rd21;
	ld.global.f32 	%f14, [%rd49];
	add.s64 	%rd50, %rd46, %rd21;
	ld.global.f32 	%f15, [%rd50];
	add.s64 	%rd51, %rd45, %rd21;
	ld.global.f32 	%f16, [%rd51];
	add.s64 	%rd52, %rd44, %rd21;
	ld.global.f32 	%f17, [%rd52];
	add.s64 	%rd53, %rd43, %rd21;
	ld.global.f32 	%f18, [%rd53];
	add.s64 	%rd1, %rd42, %rd21;
	mul.f32 	%f72, %f2, %f2;
	fma.rn.f32 	%f73, %f1, %f1, %f72;
	fma.rn.f32 	%f19, %f3, %f3, %f73;
	mul.f32 	%f393, %f1, 0f00000000;
	mul.f32 	%f394, %f2, 0f00000000;
	mul.f32 	%f395, %f3, 0f00000000;
	setp.eq.f32	%p9, %f19, 0f00000000;
	setp.eq.f32	%p10, %f383, 0f00000000;
	or.pred  	%p11, %p9, %p10;
	@%p11 bra 	BB0_34;

	ld.global.f32 	%f23, [%rd1];
	add.f32 	%f74, %f382, %f382;
	mul.f32 	%f75, %f74, %f384;
	mul.f32 	%f24, %f383, 0f40400000;
	div.rn.f32 	%f25, %f75, %f24;
	setp.gtu.f32	%p12, %f384, %f383;
	@%p12 bra 	BB0_32;
	bra.uni 	BB0_9;

BB0_32:
	sub.f32 	%f279, %f384, %f383;
	mov.f32 	%f280, 0f359EAF64;
	div.rn.f32 	%f281, %f280, %f279;
	setp.eq.f32	%p48, %f281, 0f00000000;
	div.rn.f32 	%f282, %f383, %f279;
	mul.f32 	%f283, %f19, 0f3F19999A;
	fma.rn.f32 	%f284, %f283, %f282, 0f3F800000;
	mov.f32 	%f285, 0fBF800000;
	div.rn.f32 	%f286, %f285, %f281;
	selp.f32	%f287, 0fC7C35000, %f286, %p48;
	mul.f32 	%f391, %f287, %f284;
	mov.f32 	%f392, %f25;
	bra.uni 	BB0_33;

BB0_9:
	div.rn.f32 	%f26, %f384, %f383;
	abs.f32 	%f28, %f26;
	setp.lt.f32	%p13, %f28, 0f00800000;
	mul.f32 	%f81, %f28, 0f4B800000;
	selp.f32	%f82, 0fC3170000, 0fC2FE0000, %p13;
	selp.f32	%f83, %f81, %f28, %p13;
	mov.b32 	 %r9, %f83;
	and.b32  	%r10, %r9, 8388607;
	or.b32  	%r11, %r10, 1065353216;
	mov.b32 	 %f84, %r11;
	shr.u32 	%r12, %r9, 23;
	cvt.rn.f32.u32	%f85, %r12;
	add.f32 	%f86, %f82, %f85;
	setp.gt.f32	%p14, %f84, 0f3FB504F3;
	mul.f32 	%f87, %f84, 0f3F000000;
	add.f32 	%f88, %f86, 0f3F800000;
	selp.f32	%f89, %f87, %f84, %p14;
	selp.f32	%f90, %f88, %f86, %p14;
	add.f32 	%f91, %f89, 0fBF800000;
	add.f32 	%f77, %f89, 0f3F800000;
	// inline asm
	rcp.approx.ftz.f32 %f76,%f77;
	// inline asm
	add.f32 	%f92, %f91, %f91;
	mul.f32 	%f93, %f76, %f92;
	mul.f32 	%f94, %f93, %f93;
	mov.f32 	%f95, 0f3C4CAF63;
	mov.f32 	%f96, 0f3B18F0FE;
	fma.rn.f32 	%f97, %f96, %f94, %f95;
	mov.f32 	%f98, 0f3DAAAABD;
	fma.rn.f32 	%f99, %f97, %f94, %f98;
	mul.rn.f32 	%f100, %f99, %f94;
	mul.rn.f32 	%f101, %f100, %f93;
	sub.f32 	%f102, %f91, %f93;
	neg.f32 	%f103, %f93;
	add.f32 	%f104, %f102, %f102;
	fma.rn.f32 	%f105, %f103, %f91, %f104;
	mul.rn.f32 	%f106, %f76, %f105;
	add.f32 	%f107, %f101, %f93;
	sub.f32 	%f108, %f93, %f107;
	add.f32 	%f109, %f101, %f108;
	add.f32 	%f110, %f106, %f109;
	add.f32 	%f111, %f107, %f110;
	sub.f32 	%f112, %f107, %f111;
	add.f32 	%f113, %f110, %f112;
	mov.f32 	%f114, 0f3F317200;
	mul.rn.f32 	%f115, %f90, %f114;
	mov.f32 	%f116, 0f35BFBE8E;
	mul.rn.f32 	%f117, %f90, %f116;
	add.f32 	%f118, %f115, %f111;
	sub.f32 	%f119, %f115, %f118;
	add.f32 	%f120, %f111, %f119;
	add.f32 	%f121, %f113, %f120;
	add.f32 	%f122, %f117, %f121;
	add.f32 	%f123, %f118, %f122;
	sub.f32 	%f124, %f118, %f123;
	add.f32 	%f125, %f122, %f124;
	mov.f32 	%f126, 0f405F5C29;
	mul.rn.f32 	%f127, %f126, %f123;
	neg.f32 	%f128, %f127;
	fma.rn.f32 	%f129, %f126, %f123, %f128;
	fma.rn.f32 	%f130, %f126, %f125, %f129;
	mov.f32 	%f131, 0f00000000;
	fma.rn.f32 	%f132, %f131, %f123, %f130;
	add.rn.f32 	%f133, %f127, %f132;
	neg.f32 	%f134, %f133;
	add.rn.f32 	%f135, %f127, %f134;
	add.rn.f32 	%f136, %f135, %f132;
	mov.b32 	 %r13, %f133;
	setp.eq.s32	%p15, %r13, 1118925336;
	add.s32 	%r14, %r13, -1;
	mov.b32 	 %f137, %r14;
	add.f32 	%f138, %f136, 0f37000000;
	selp.f32	%f139, %f137, %f133, %p15;
	selp.f32	%f29, %f138, %f136, %p15;
	mul.f32 	%f140, %f139, 0f3FB8AA3B;
	cvt.rzi.f32.f32	%f141, %f140;
	mov.f32 	%f142, 0fBF317200;
	fma.rn.f32 	%f143, %f141, %f142, %f139;
	mov.f32 	%f144, 0fB5BFBE8E;
	fma.rn.f32 	%f145, %f141, %f144, %f143;
	mul.f32 	%f146, %f145, 0f3FB8AA3B;
	ex2.approx.ftz.f32 	%f147, %f146;
	add.f32 	%f148, %f141, 0f00000000;
	ex2.approx.f32 	%f149, %f148;
	mul.f32 	%f150, %f147, %f149;
	setp.lt.f32	%p16, %f139, 0fC2D20000;
	selp.f32	%f151, 0f00000000, %f150, %p16;
	setp.gt.f32	%p17, %f139, 0f42D20000;
	selp.f32	%f385, 0f7F800000, %f151, %p17;
	setp.eq.f32	%p18, %f385, 0f7F800000;
	@%p18 bra 	BB0_11;

	fma.rn.f32 	%f385, %f385, %f29, %f385;

BB0_11:
	mov.f32 	%f378, 0f3FDF5C29;
	cvt.rzi.f32.f32	%f377, %f378;
	fma.rn.f32 	%f376, %f377, 0fC0000000, 0f405F5C29;
	abs.f32 	%f375, %f376;
	setp.lt.f32	%p19, %f26, 0f00000000;
	setp.eq.f32	%p20, %f375, 0f3F800000;
	and.pred  	%p1, %p19, %p20;
	mov.b32 	 %r15, %f385;
	xor.b32  	%r16, %r15, -2147483648;
	mov.b32 	 %f152, %r16;
	selp.f32	%f387, %f152, %f385, %p1;
	setp.eq.f32	%p21, %f26, 0f00000000;
	@%p21 bra 	BB0_14;
	bra.uni 	BB0_12;

BB0_14:
	add.f32 	%f155, %f26, %f26;
	selp.f32	%f387, %f155, 0f00000000, %p20;
	bra.uni 	BB0_15;

BB0_12:
	setp.geu.f32	%p22, %f26, 0f00000000;
	@%p22 bra 	BB0_15;

	mov.f32 	%f364, 0f405F5C29;
	cvt.rzi.f32.f32	%f154, %f364;
	setp.neu.f32	%p23, %f154, 0f405F5C29;
	selp.f32	%f387, 0f7FFFFFFF, %f387, %p23;

BB0_15:
	abs.f32 	%f379, %f26;
	add.f32 	%f156, %f379, 0f405F5C29;
	mov.b32 	 %r17, %f156;
	setp.lt.s32	%p25, %r17, 2139095040;
	@%p25 bra 	BB0_20;

	abs.f32 	%f380, %f26;
	setp.gtu.f32	%p26, %f380, 0f7F800000;
	@%p26 bra 	BB0_19;
	bra.uni 	BB0_17;

BB0_19:
	add.f32 	%f387, %f26, 0f405F5C29;
	bra.uni 	BB0_20;

BB0_17:
	abs.f32 	%f381, %f26;
	setp.neu.f32	%p27, %f381, 0f7F800000;
	@%p27 bra 	BB0_20;

	selp.f32	%f387, 0fFF800000, 0f7F800000, %p1;

BB0_20:
	mov.f32 	%f361, 0fB5BFBE8E;
	mov.f32 	%f360, 0fBF317200;
	mov.f32 	%f359, 0f00000000;
	mov.f32 	%f358, 0f35BFBE8E;
	mov.f32 	%f357, 0f3F317200;
	mov.f32 	%f356, 0f3DAAAABD;
	mov.f32 	%f355, 0f3C4CAF63;
	mov.f32 	%f354, 0f3B18F0FE;
	mov.f32 	%f159, 0f3F800000;
	sub.f32 	%f160, %f159, %f387;
	setp.eq.f32	%p28, %f26, 0f3F800000;
	selp.f32	%f40, 0f00000000, %f160, %p28;
	abs.f32 	%f42, %f40;
	setp.lt.f32	%p29, %f42, 0f00800000;
	mul.f32 	%f164, %f42, 0f4B800000;
	selp.f32	%f165, 0fC3170000, 0fC2FE0000, %p29;
	selp.f32	%f166, %f164, %f42, %p29;
	mov.b32 	 %r18, %f166;
	and.b32  	%r19, %r18, 8388607;
	or.b32  	%r20, %r19, 1065353216;
	mov.b32 	 %f167, %r20;
	shr.u32 	%r21, %r18, 23;
	cvt.rn.f32.u32	%f168, %r21;
	add.f32 	%f169, %f165, %f168;
	setp.gt.f32	%p30, %f167, 0f3FB504F3;
	mul.f32 	%f170, %f167, 0f3F000000;
	add.f32 	%f171, %f169, 0f3F800000;
	selp.f32	%f172, %f170, %f167, %p30;
	selp.f32	%f173, %f171, %f169, %p30;
	add.f32 	%f174, %f172, 0fBF800000;
	add.f32 	%f158, %f172, 0f3F800000;
	// inline asm
	rcp.approx.ftz.f32 %f157,%f158;
	// inline asm
	add.f32 	%f175, %f174, %f174;
	mul.f32 	%f176, %f157, %f175;
	mul.f32 	%f177, %f176, %f176;
	fma.rn.f32 	%f180, %f354, %f177, %f355;
	fma.rn.f32 	%f182, %f180, %f177, %f356;
	mul.rn.f32 	%f183, %f182, %f177;
	mul.rn.f32 	%f184, %f183, %f176;
	sub.f32 	%f185, %f174, %f176;
	neg.f32 	%f186, %f176;
	add.f32 	%f187, %f185, %f185;
	fma.rn.f32 	%f188, %f186, %f174, %f187;
	mul.rn.f32 	%f189, %f157, %f188;
	add.f32 	%f190, %f184, %f176;
	sub.f32 	%f191, %f176, %f190;
	add.f32 	%f192, %f184, %f191;
	add.f32 	%f193, %f189, %f192;
	add.f32 	%f194, %f190, %f193;
	sub.f32 	%f195, %f190, %f194;
	add.f32 	%f196, %f193, %f195;
	mul.rn.f32 	%f198, %f173, %f357;
	mul.rn.f32 	%f200, %f173, %f358;
	add.f32 	%f201, %f198, %f194;
	sub.f32 	%f202, %f198, %f201;
	add.f32 	%f203, %f194, %f202;
	add.f32 	%f204, %f196, %f203;
	add.f32 	%f205, %f200, %f204;
	add.f32 	%f206, %f201, %f205;
	sub.f32 	%f207, %f201, %f206;
	add.f32 	%f208, %f205, %f207;
	mov.f32 	%f209, 0f3F0A3D71;
	mul.rn.f32 	%f210, %f209, %f206;
	neg.f32 	%f211, %f210;
	fma.rn.f32 	%f212, %f209, %f206, %f211;
	fma.rn.f32 	%f213, %f209, %f208, %f212;
	fma.rn.f32 	%f215, %f359, %f206, %f213;
	add.rn.f32 	%f216, %f210, %f215;
	neg.f32 	%f217, %f216;
	add.rn.f32 	%f218, %f210, %f217;
	add.rn.f32 	%f219, %f218, %f215;
	mov.b32 	 %r22, %f216;
	setp.eq.s32	%p31, %r22, 1118925336;
	add.s32 	%r23, %r22, -1;
	mov.b32 	 %f220, %r23;
	add.f32 	%f221, %f219, 0f37000000;
	selp.f32	%f222, %f220, %f216, %p31;
	selp.f32	%f43, %f221, %f219, %p31;
	mul.f32 	%f223, %f222, 0f3FB8AA3B;
	cvt.rzi.f32.f32	%f224, %f223;
	fma.rn.f32 	%f226, %f224, %f360, %f222;
	fma.rn.f32 	%f228, %f224, %f361, %f226;
	mul.f32 	%f229, %f228, 0f3FB8AA3B;
	ex2.approx.ftz.f32 	%f230, %f229;
	add.f32 	%f231, %f224, 0f00000000;
	ex2.approx.f32 	%f232, %f231;
	mul.f32 	%f233, %f230, %f232;
	setp.lt.f32	%p32, %f222, 0fC2D20000;
	selp.f32	%f234, 0f00000000, %f233, %p32;
	setp.gt.f32	%p33, %f222, 0f42D20000;
	selp.f32	%f388, 0f7F800000, %f234, %p33;
	setp.eq.f32	%p34, %f388, 0f7F800000;
	@%p34 bra 	BB0_22;

	fma.rn.f32 	%f388, %f388, %f43, %f388;

BB0_22:
	mov.f32 	%f368, 0f3E8A3D71;
	cvt.rzi.f32.f32	%f367, %f368;
	fma.rn.f32 	%f366, %f367, 0fC0000000, 0f3F0A3D71;
	abs.f32 	%f365, %f366;
	setp.lt.f32	%p35, %f40, 0f00000000;
	setp.eq.f32	%p36, %f365, 0f3F800000;
	and.pred  	%p2, %p35, %p36;
	mov.b32 	 %r24, %f388;
	xor.b32  	%r25, %r24, -2147483648;
	mov.b32 	 %f235, %r25;
	selp.f32	%f390, %f235, %f388, %p2;
	setp.eq.f32	%p37, %f40, 0f00000000;
	@%p37 bra 	BB0_25;
	bra.uni 	BB0_23;

BB0_25:
	add.f32 	%f238, %f40, %f40;
	selp.f32	%f390, %f238, 0f00000000, %p36;
	bra.uni 	BB0_26;

BB0_23:
	setp.geu.f32	%p38, %f40, 0f00000000;
	@%p38 bra 	BB0_26;

	mov.f32 	%f374, 0f3F0A3D71;
	cvt.rzi.f32.f32	%f237, %f374;
	setp.neu.f32	%p39, %f237, 0f3F0A3D71;
	selp.f32	%f390, 0f7FFFFFFF, %f390, %p39;

BB0_26:
	abs.f32 	%f369, %f40;
	add.f32 	%f239, %f369, 0f3F0A3D71;
	mov.b32 	 %r26, %f239;
	setp.lt.s32	%p41, %r26, 2139095040;
	@%p41 bra 	BB0_31;

	abs.f32 	%f372, %f40;
	setp.gtu.f32	%p42, %f372, 0f7F800000;
	@%p42 bra 	BB0_30;
	bra.uni 	BB0_28;

BB0_30:
	add.f32 	%f390, %f40, 0f3F0A3D71;
	bra.uni 	BB0_31;

BB0_28:
	abs.f32 	%f373, %f40;
	setp.neu.f32	%p43, %f373, 0f7F800000;
	@%p43 bra 	BB0_31;

	selp.f32	%f390, 0fFF800000, 0f7F800000, %p2;

BB0_31:
	mul.f32 	%f371, %f383, 0f40400000;
	mov.f32 	%f370, 0f3F800000;
	mov.f32 	%f363, 0fB5BFBE8E;
	mov.f32 	%f362, 0fBF317200;
	setp.eq.f32	%p44, %f40, 0f3F800000;
	selp.f32	%f240, 0f3F800000, %f390, %p44;
	cvt.f64.f32	%fd5, %f240;
	setp.lt.f64	%p45, %fd5, 0d3F50624DD2F1A9FC;
	selp.f32	%f241, 0f3A83126F, %f240, %p45;
	mul.f32 	%f242, %f384, 0f19857725;
	rcp.rn.f32 	%f243, %f242;
	div.rn.f32 	%f244, %f383, %f384;
	mul.f32 	%f245, %f244, %f241;
	abs.f32 	%f246, %f245;
	mul.f32 	%f247, %f246, 0f3FB8AA3B;
	cvt.rzi.f32.f32	%f248, %f247;
	fma.rn.f32 	%f250, %f248, %f362, %f246;
	fma.rn.f32 	%f252, %f248, %f363, %f250;
	mul.f32 	%f253, %f252, 0f3FB8AA3B;
	ex2.approx.ftz.f32 	%f254, %f253;
	add.f32 	%f255, %f248, 0fC0000000;
	ex2.approx.f32 	%f256, %f255;
	mul.f32 	%f257, %f254, %f256;
	mov.f32 	%f258, 0f3E000000;
	div.approx.f32 	%f259, %f258, %f257;
	mov.f32 	%f260, 0f40000000;
	fma.rn.f32 	%f261, %f260, %f257, %f259;
	setp.ltu.f32	%p46, %f246, 0f42B40000;
	selp.f32	%f262, %f261, 0f7F800000, %p46;
	mul.f32 	%f263, %f262, %f262;
	rcp.rn.f32 	%f264, %f263;
	mul.f32 	%f265, %f243, 0f0FA575F3;
	mul.f32 	%f266, %f244, %f264;
	sub.f32 	%f268, %f370, %f266;
	div.rn.f32 	%f269, %f264, %f268;
	mul.f32 	%f270, %f265, %f269;
	cvt.f64.f32	%fd6, %f270;
	setp.lt.f64	%p47, %fd6, 0d3DD5FD7FE1796495;
	div.rn.f32 	%f271, %f384, %f371;
	sub.f32 	%f272, %f370, %f271;
	mul.f32 	%f392, %f382, %f272;
	add.f32 	%f273, %f270, %f270;
	mul.f32 	%f274, %f241, %f241;
	div.rn.f32 	%f275, %f19, %f274;
	sub.f32 	%f276, %f370, %f275;
	rcp.rn.f32 	%f277, %f273;
	selp.f32	%f278, 0f4FBA43B7, %f277, %p47;
	mul.f32 	%f391, %f276, %f278;

BB0_33:
	mul.f32 	%f288, %f391, 0f35A8A9B8;
	mul.f32 	%f289, %f382, %f392;
	mul.f32 	%f290, %f392, %f289;
	sub.f32 	%f291, %f392, %f25;
	div.rn.f32 	%f292, %f291, %f290;
	sqrt.rn.f32 	%f293, %f292;
	div.rn.f32 	%f294, %f25, %f382;
	sqrt.rn.f32 	%f295, %f294;
	fma.rn.f32 	%f296, %f1, %f288, %f4;
	fma.rn.f32 	%f297, %f2, %f288, %f5;
	fma.rn.f32 	%f298, %f3, %f288, %f6;
	fma.rn.f32 	%f299, %f14, %f293, %f296;
	fma.rn.f32 	%f300, %f15, %f293, %f297;
	fma.rn.f32 	%f301, %f16, %f293, %f298;
	mul.f32 	%f302, %f2, %f298;
	mul.f32 	%f303, %f3, %f297;
	sub.f32 	%f304, %f302, %f303;
	mul.f32 	%f305, %f3, %f296;
	mul.f32 	%f306, %f1, %f298;
	sub.f32 	%f307, %f305, %f306;
	mul.f32 	%f308, %f1, %f297;
	mul.f32 	%f309, %f2, %f296;
	sub.f32 	%f310, %f308, %f309;
	mul.f32 	%f311, %f2, %f297;
	fma.rn.f32 	%f312, %f1, %f296, %f311;
	fma.rn.f32 	%f313, %f3, %f298, %f312;
	mul.f32 	%f314, %f2, %f301;
	mul.f32 	%f315, %f3, %f300;
	sub.f32 	%f316, %f314, %f315;
	mul.f32 	%f317, %f3, %f299;
	mul.f32 	%f318, %f1, %f301;
	sub.f32 	%f319, %f317, %f318;
	mul.f32 	%f320, %f1, %f300;
	mul.f32 	%f321, %f2, %f299;
	sub.f32 	%f322, %f320, %f321;
	mul.f32 	%f323, %f2, %f322;
	mul.f32 	%f324, %f3, %f319;
	sub.f32 	%f325, %f323, %f324;
	mul.f32 	%f326, %f3, %f316;
	mul.f32 	%f327, %f1, %f322;
	sub.f32 	%f328, %f326, %f327;
	mul.f32 	%f329, %f1, %f319;
	mul.f32 	%f330, %f2, %f316;
	sub.f32 	%f331, %f329, %f330;
	fma.rn.f32 	%f332, %f382, %f382, 0f3F800000;
	rcp.rn.f32 	%f333, %f332;
	mul.f32 	%f334, %f304, %f333;
	mul.f32 	%f335, %f307, %f333;
	mul.f32 	%f336, %f310, %f333;
	mul.f32 	%f337, %f25, %f333;
	div.rn.f32 	%f338, %f337, %f19;
	mul.f32 	%f339, %f313, %f338;
	mul.f32 	%f340, %f1, %f339;
	mul.f32 	%f341, %f2, %f339;
	mul.f32 	%f342, %f3, %f339;
	sub.f32 	%f343, %f340, %f334;
	sub.f32 	%f344, %f341, %f335;
	sub.f32 	%f345, %f342, %f336;
	mul.f32 	%f346, %f392, %f333;
	div.rn.f32 	%f347, %f346, %f19;
	mul.f32 	%f348, %f347, %f325;
	mul.f32 	%f349, %f347, %f328;
	mul.f32 	%f350, %f347, %f331;
	sub.f32 	%f351, %f343, %f348;
	sub.f32 	%f352, %f344, %f349;
	sub.f32 	%f353, %f345, %f350;
	fma.rn.f32 	%f393, %f17, %f295, %f351;
	fma.rn.f32 	%f394, %f18, %f295, %f352;
	fma.rn.f32 	%f395, %f23, %f295, %f353;

BB0_34:
	ld.param.u64 	%rd64, [LLBtorque2JH_param_2];
	ld.param.u64 	%rd63, [LLBtorque2JH_param_1];
	ld.param.u64 	%rd62, [LLBtorque2JH_param_0];
	mov.u32 	%r40, %ctaid.x;
	mov.u32 	%r39, %ctaid.y;
	mov.u32 	%r38, %nctaid.x;
	mov.u32 	%r37, %tid.x;
	mov.u32 	%r36, %ntid.x;
	mad.lo.s32 	%r35, %r38, %r39, %r40;
	mad.lo.s32 	%r34, %r35, %r36, %r37;
	mul.wide.s32 	%rd61, %r34, 4;
	cvta.to.global.u64 	%rd54, %rd62;
	add.s64 	%rd56, %rd54, %rd61;
	st.global.f32 	[%rd56], %f393;
	cvta.to.global.u64 	%rd57, %rd63;
	add.s64 	%rd58, %rd57, %rd61;
	st.global.f32 	[%rd58], %f394;
	cvta.to.global.u64 	%rd59, %rd64;
	add.s64 	%rd60, %rd59, %rd61;
	st.global.f32 	[%rd60], %f395;

BB0_35:
	ret;
}


`
)
