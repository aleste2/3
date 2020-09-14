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

// CUDA handle for addexchangeAfll kernel
var addexchangeAfll_code cu.Function

// Stores the arguments for addexchangeAfll kernel invocation
type addexchangeAfll_args_t struct {
	arg_dst1x   unsafe.Pointer
	arg_dst1y   unsafe.Pointer
	arg_dst1z   unsafe.Pointer
	arg_dst2x   unsafe.Pointer
	arg_dst2y   unsafe.Pointer
	arg_dst2z   unsafe.Pointer
	arg_m1x     unsafe.Pointer
	arg_m1y     unsafe.Pointer
	arg_m1z     unsafe.Pointer
	arg_m2x     unsafe.Pointer
	arg_m2y     unsafe.Pointer
	arg_m2z     unsafe.Pointer
	arg_Ms1_    unsafe.Pointer
	arg_Ms1_mul float32
	arg_Ms2_    unsafe.Pointer
	arg_Ms2_mul float32
	arg_aLUT2d  unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_wx      float32
	arg_wy      float32
	arg_wz      float32
	arg_Nx      int
	arg_Ny      int
	arg_Nz      int
	arg_PBC     byte
	argptr      [25]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for addexchangeAfll kernel invocation
var addexchangeAfll_args addexchangeAfll_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	addexchangeAfll_args.argptr[0] = unsafe.Pointer(&addexchangeAfll_args.arg_dst1x)
	addexchangeAfll_args.argptr[1] = unsafe.Pointer(&addexchangeAfll_args.arg_dst1y)
	addexchangeAfll_args.argptr[2] = unsafe.Pointer(&addexchangeAfll_args.arg_dst1z)
	addexchangeAfll_args.argptr[3] = unsafe.Pointer(&addexchangeAfll_args.arg_dst2x)
	addexchangeAfll_args.argptr[4] = unsafe.Pointer(&addexchangeAfll_args.arg_dst2y)
	addexchangeAfll_args.argptr[5] = unsafe.Pointer(&addexchangeAfll_args.arg_dst2z)
	addexchangeAfll_args.argptr[6] = unsafe.Pointer(&addexchangeAfll_args.arg_m1x)
	addexchangeAfll_args.argptr[7] = unsafe.Pointer(&addexchangeAfll_args.arg_m1y)
	addexchangeAfll_args.argptr[8] = unsafe.Pointer(&addexchangeAfll_args.arg_m1z)
	addexchangeAfll_args.argptr[9] = unsafe.Pointer(&addexchangeAfll_args.arg_m2x)
	addexchangeAfll_args.argptr[10] = unsafe.Pointer(&addexchangeAfll_args.arg_m2y)
	addexchangeAfll_args.argptr[11] = unsafe.Pointer(&addexchangeAfll_args.arg_m2z)
	addexchangeAfll_args.argptr[12] = unsafe.Pointer(&addexchangeAfll_args.arg_Ms1_)
	addexchangeAfll_args.argptr[13] = unsafe.Pointer(&addexchangeAfll_args.arg_Ms1_mul)
	addexchangeAfll_args.argptr[14] = unsafe.Pointer(&addexchangeAfll_args.arg_Ms2_)
	addexchangeAfll_args.argptr[15] = unsafe.Pointer(&addexchangeAfll_args.arg_Ms2_mul)
	addexchangeAfll_args.argptr[16] = unsafe.Pointer(&addexchangeAfll_args.arg_aLUT2d)
	addexchangeAfll_args.argptr[17] = unsafe.Pointer(&addexchangeAfll_args.arg_regions)
	addexchangeAfll_args.argptr[18] = unsafe.Pointer(&addexchangeAfll_args.arg_wx)
	addexchangeAfll_args.argptr[19] = unsafe.Pointer(&addexchangeAfll_args.arg_wy)
	addexchangeAfll_args.argptr[20] = unsafe.Pointer(&addexchangeAfll_args.arg_wz)
	addexchangeAfll_args.argptr[21] = unsafe.Pointer(&addexchangeAfll_args.arg_Nx)
	addexchangeAfll_args.argptr[22] = unsafe.Pointer(&addexchangeAfll_args.arg_Ny)
	addexchangeAfll_args.argptr[23] = unsafe.Pointer(&addexchangeAfll_args.arg_Nz)
	addexchangeAfll_args.argptr[24] = unsafe.Pointer(&addexchangeAfll_args.arg_PBC)
}

// Wrapper for addexchangeAfll CUDA kernel, asynchronous.
func k_addexchangeAfll_async(dst1x unsafe.Pointer, dst1y unsafe.Pointer, dst1z unsafe.Pointer, dst2x unsafe.Pointer, dst2y unsafe.Pointer, dst2z unsafe.Pointer, m1x unsafe.Pointer, m1y unsafe.Pointer, m1z unsafe.Pointer, m2x unsafe.Pointer, m2y unsafe.Pointer, m2z unsafe.Pointer, Ms1_ unsafe.Pointer, Ms1_mul float32, Ms2_ unsafe.Pointer, Ms2_mul float32, aLUT2d unsafe.Pointer, regions unsafe.Pointer, wx float32, wy float32, wz float32, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("addexchangeAfll")
	}

	addexchangeAfll_args.Lock()
	defer addexchangeAfll_args.Unlock()

	if addexchangeAfll_code == 0 {
		addexchangeAfll_code = fatbinLoad(addexchangeAfll_map, "addexchangeAfll")
	}

	addexchangeAfll_args.arg_dst1x = dst1x
	addexchangeAfll_args.arg_dst1y = dst1y
	addexchangeAfll_args.arg_dst1z = dst1z
	addexchangeAfll_args.arg_dst2x = dst2x
	addexchangeAfll_args.arg_dst2y = dst2y
	addexchangeAfll_args.arg_dst2z = dst2z
	addexchangeAfll_args.arg_m1x = m1x
	addexchangeAfll_args.arg_m1y = m1y
	addexchangeAfll_args.arg_m1z = m1z
	addexchangeAfll_args.arg_m2x = m2x
	addexchangeAfll_args.arg_m2y = m2y
	addexchangeAfll_args.arg_m2z = m2z
	addexchangeAfll_args.arg_Ms1_ = Ms1_
	addexchangeAfll_args.arg_Ms1_mul = Ms1_mul
	addexchangeAfll_args.arg_Ms2_ = Ms2_
	addexchangeAfll_args.arg_Ms2_mul = Ms2_mul
	addexchangeAfll_args.arg_aLUT2d = aLUT2d
	addexchangeAfll_args.arg_regions = regions
	addexchangeAfll_args.arg_wx = wx
	addexchangeAfll_args.arg_wy = wy
	addexchangeAfll_args.arg_wz = wz
	addexchangeAfll_args.arg_Nx = Nx
	addexchangeAfll_args.arg_Ny = Ny
	addexchangeAfll_args.arg_Nz = Nz
	addexchangeAfll_args.arg_PBC = PBC

	args := addexchangeAfll_args.argptr[:]
	cu.LaunchKernel(addexchangeAfll_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("addexchangeAfll")
	}
}

// maps compute capability on PTX code for addexchangeAfll kernel.
var addexchangeAfll_map = map[int]string{0: "",
	30: addexchangeAfll_ptx_30}

// addexchangeAfll PTX code for various compute capabilities.
const (
	addexchangeAfll_ptx_30 = `
.version 6.5
.target sm_30
.address_size 64

	// .globl	addexchangeAfll

.visible .entry addexchangeAfll(
	.param .u64 addexchangeAfll_param_0,
	.param .u64 addexchangeAfll_param_1,
	.param .u64 addexchangeAfll_param_2,
	.param .u64 addexchangeAfll_param_3,
	.param .u64 addexchangeAfll_param_4,
	.param .u64 addexchangeAfll_param_5,
	.param .u64 addexchangeAfll_param_6,
	.param .u64 addexchangeAfll_param_7,
	.param .u64 addexchangeAfll_param_8,
	.param .u64 addexchangeAfll_param_9,
	.param .u64 addexchangeAfll_param_10,
	.param .u64 addexchangeAfll_param_11,
	.param .u64 addexchangeAfll_param_12,
	.param .f32 addexchangeAfll_param_13,
	.param .u64 addexchangeAfll_param_14,
	.param .f32 addexchangeAfll_param_15,
	.param .u64 addexchangeAfll_param_16,
	.param .u64 addexchangeAfll_param_17,
	.param .f32 addexchangeAfll_param_18,
	.param .f32 addexchangeAfll_param_19,
	.param .f32 addexchangeAfll_param_20,
	.param .u32 addexchangeAfll_param_21,
	.param .u32 addexchangeAfll_param_22,
	.param .u32 addexchangeAfll_param_23,
	.param .u8 addexchangeAfll_param_24
)
{
	.reg .pred 	%p<28>;
	.reg .b16 	%rs<21>;
	.reg .f32 	%f<133>;
	.reg .b32 	%r<247>;
	.reg .b64 	%rd<108>;


	ld.param.u64 	%rd1, [addexchangeAfll_param_0];
	ld.param.u64 	%rd2, [addexchangeAfll_param_1];
	ld.param.u64 	%rd3, [addexchangeAfll_param_2];
	ld.param.u64 	%rd4, [addexchangeAfll_param_9];
	ld.param.u64 	%rd5, [addexchangeAfll_param_10];
	ld.param.u64 	%rd6, [addexchangeAfll_param_11];
	ld.param.u64 	%rd7, [addexchangeAfll_param_12];
	ld.param.f32 	%f128, [addexchangeAfll_param_13];
	ld.param.u64 	%rd8, [addexchangeAfll_param_16];
	ld.param.u64 	%rd9, [addexchangeAfll_param_17];
	ld.param.f32 	%f30, [addexchangeAfll_param_18];
	ld.param.f32 	%f31, [addexchangeAfll_param_19];
	ld.param.f32 	%f32, [addexchangeAfll_param_20];
	ld.param.u32 	%r27, [addexchangeAfll_param_21];
	ld.param.u32 	%r28, [addexchangeAfll_param_22];
	ld.param.u32 	%r29, [addexchangeAfll_param_23];
	ld.param.u8 	%rs4, [addexchangeAfll_param_24];
	mov.u32 	%r30, %ntid.x;
	mov.u32 	%r31, %ctaid.x;
	mov.u32 	%r32, %tid.x;
	mad.lo.s32 	%r1, %r30, %r31, %r32;
	mov.u32 	%r33, %ntid.y;
	mov.u32 	%r34, %ctaid.y;
	mov.u32 	%r35, %tid.y;
	mad.lo.s32 	%r2, %r33, %r34, %r35;
	mov.u32 	%r36, %ntid.z;
	mov.u32 	%r37, %ctaid.z;
	mov.u32 	%r38, %tid.z;
	mad.lo.s32 	%r3, %r36, %r37, %r38;
	setp.ge.s32	%p1, %r2, %r28;
	setp.ge.s32	%p2, %r1, %r27;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r29;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_27;

	cvta.to.global.u64 	%rd10, %rd4;
	mad.lo.s32 	%r39, %r3, %r28, %r2;
	mad.lo.s32 	%r40, %r39, %r27, %r1;
	mul.wide.s32 	%rd11, %r40, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.f32 	%f1, [%rd12];
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd14, %rd13, %rd11;
	ld.global.f32 	%f2, [%rd14];
	cvta.to.global.u64 	%rd15, %rd6;
	add.s64 	%rd16, %rd15, %rd11;
	ld.global.f32 	%f3, [%rd16];
	setp.eq.s64	%p6, %rd7, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd17, %rd7;
	add.s64 	%rd19, %rd17, %rd11;
	ld.global.f32 	%f33, [%rd19];
	mul.f32 	%f128, %f33, %f128;

BB0_3:
	setp.eq.f32	%p7, %f128, 0f00000000;
	mov.f32 	%f129, 0f00000000;
	@%p7 bra 	BB0_5;

	rcp.rn.f32 	%f129, %f128;

BB0_5:
	mul.f32 	%f35, %f2, %f2;
	fma.rn.f32 	%f36, %f1, %f1, %f35;
	fma.rn.f32 	%f37, %f3, %f3, %f36;
	setp.eq.f32	%p8, %f37, 0f00000000;
	@%p8 bra 	BB0_27;

	cvta.to.global.u64 	%rd20, %rd9;
	cvt.s64.s32	%rd21, %r40;
	add.s64 	%rd22, %rd20, %rd21;
	ld.global.u8 	%rs1, [%rd22];
	and.b16  	%rs5, %rs4, 1;
	setp.eq.b16	%p9, %rs5, 1;
	@!%p9 bra 	BB0_8;
	bra.uni 	BB0_7;

BB0_7:
	add.s32 	%r73, %r1, -1;
	rem.s32 	%r74, %r73, %r27;
	add.s32 	%r75, %r74, %r27;
	rem.s32 	%r241, %r75, %r27;
	bra.uni 	BB0_9;

BB0_8:
	add.s32 	%r80, %r1, -1;
	mov.u32 	%r81, 0;
	max.s32 	%r241, %r80, %r81;

BB0_9:
	setp.eq.b16	%p10, %rs5, 1;
	mad.lo.s32 	%r91, %r39, %r27, %r241;
	cvt.s64.s32	%rd24, %r91;
	mul.wide.s32 	%rd25, %r91, 4;
	add.s64 	%rd26, %rd10, %rd25;
	add.s64 	%rd28, %rd13, %rd25;
	add.s64 	%rd30, %rd15, %rd25;
	ld.global.f32 	%f38, [%rd26];
	ld.global.f32 	%f39, [%rd28];
	mul.f32 	%f40, %f39, %f39;
	fma.rn.f32 	%f41, %f38, %f38, %f40;
	ld.global.f32 	%f42, [%rd30];
	fma.rn.f32 	%f43, %f42, %f42, %f41;
	setp.eq.f32	%p11, %f43, 0f00000000;
	selp.f32	%f44, %f1, %f38, %p11;
	selp.f32	%f45, %f2, %f39, %p11;
	selp.f32	%f46, %f3, %f42, %p11;
	add.s64 	%rd32, %rd20, %rd24;
	ld.global.u8 	%rs7, [%rd32];
	setp.gt.u16	%p12, %rs7, %rs1;
	cvt.u32.u16	%r92, %rs7;
	cvt.u32.u16	%r93, %rs1;
	and.b32  	%r94, %r93, 255;
	selp.b32	%r95, %r94, %r92, %p12;
	selp.b32	%r96, %r92, %r94, %p12;
	add.s32 	%r97, %r96, 1;
	mul.lo.s32 	%r98, %r97, %r96;
	shr.u32 	%r99, %r98, 1;
	add.s32 	%r100, %r99, %r95;
	cvta.to.global.u64 	%rd33, %rd8;
	mul.wide.s32 	%rd34, %r100, 4;
	add.s64 	%rd35, %rd33, %rd34;
	ld.global.f32 	%f47, [%rd35];
	mul.f32 	%f48, %f47, %f30;
	sub.f32 	%f49, %f44, %f1;
	sub.f32 	%f50, %f45, %f2;
	sub.f32 	%f51, %f46, %f3;
	fma.rn.f32 	%f8, %f48, %f49, 0f00000000;
	fma.rn.f32 	%f9, %f48, %f50, 0f00000000;
	fma.rn.f32 	%f10, %f48, %f51, 0f00000000;
	add.s32 	%r7, %r1, 1;
	@!%p10 bra 	BB0_11;
	bra.uni 	BB0_10;

BB0_10:
	rem.s32 	%r105, %r7, %r27;
	add.s32 	%r106, %r105, %r27;
	rem.s32 	%r242, %r106, %r27;
	bra.uni 	BB0_12;

BB0_11:
	add.s32 	%r107, %r27, -1;
	min.s32 	%r242, %r7, %r107;

BB0_12:
	mad.lo.s32 	%r117, %r39, %r27, %r242;
	cvt.s64.s32	%rd37, %r117;
	mul.wide.s32 	%rd38, %r117, 4;
	add.s64 	%rd39, %rd10, %rd38;
	add.s64 	%rd41, %rd13, %rd38;
	add.s64 	%rd43, %rd15, %rd38;
	ld.global.f32 	%f52, [%rd39];
	ld.global.f32 	%f53, [%rd41];
	mul.f32 	%f54, %f53, %f53;
	fma.rn.f32 	%f55, %f52, %f52, %f54;
	ld.global.f32 	%f56, [%rd43];
	fma.rn.f32 	%f57, %f56, %f56, %f55;
	setp.eq.f32	%p13, %f57, 0f00000000;
	selp.f32	%f58, %f1, %f52, %p13;
	selp.f32	%f59, %f2, %f53, %p13;
	selp.f32	%f60, %f3, %f56, %p13;
	add.s64 	%rd45, %rd20, %rd37;
	ld.global.u8 	%rs9, [%rd45];
	setp.gt.u16	%p14, %rs9, %rs1;
	cvt.u32.u16	%r118, %rs9;
	selp.b32	%r121, %r94, %r118, %p14;
	selp.b32	%r122, %r118, %r94, %p14;
	add.s32 	%r123, %r122, 1;
	mul.lo.s32 	%r124, %r123, %r122;
	shr.u32 	%r125, %r124, 1;
	add.s32 	%r126, %r125, %r121;
	mul.wide.s32 	%rd47, %r126, 4;
	add.s64 	%rd48, %rd33, %rd47;
	ld.global.f32 	%f61, [%rd48];
	mul.f32 	%f62, %f61, %f30;
	sub.f32 	%f63, %f58, %f1;
	sub.f32 	%f64, %f59, %f2;
	sub.f32 	%f65, %f60, %f3;
	fma.rn.f32 	%f11, %f62, %f63, %f8;
	fma.rn.f32 	%f12, %f62, %f64, %f9;
	fma.rn.f32 	%f13, %f62, %f65, %f10;
	and.b16  	%rs2, %rs4, 2;
	setp.eq.s16	%p15, %rs2, 0;
	add.s32 	%r11, %r2, -1;
	@%p15 bra 	BB0_14;

	rem.s32 	%r127, %r11, %r28;
	add.s32 	%r128, %r127, %r28;
	rem.s32 	%r243, %r128, %r28;
	bra.uni 	BB0_15;

BB0_14:
	mov.u32 	%r129, 0;
	max.s32 	%r243, %r11, %r129;

BB0_15:
	mad.lo.s32 	%r134, %r3, %r28, %r243;
	mad.lo.s32 	%r139, %r134, %r27, %r1;
	cvt.s64.s32	%rd50, %r139;
	mul.wide.s32 	%rd51, %r139, 4;
	add.s64 	%rd52, %rd10, %rd51;
	add.s64 	%rd54, %rd13, %rd51;
	add.s64 	%rd56, %rd15, %rd51;
	ld.global.f32 	%f66, [%rd52];
	ld.global.f32 	%f67, [%rd54];
	mul.f32 	%f68, %f67, %f67;
	fma.rn.f32 	%f69, %f66, %f66, %f68;
	ld.global.f32 	%f70, [%rd56];
	fma.rn.f32 	%f71, %f70, %f70, %f69;
	setp.eq.f32	%p16, %f71, 0f00000000;
	selp.f32	%f72, %f1, %f66, %p16;
	selp.f32	%f73, %f2, %f67, %p16;
	selp.f32	%f74, %f3, %f70, %p16;
	add.s64 	%rd58, %rd20, %rd50;
	ld.global.u8 	%rs11, [%rd58];
	setp.gt.u16	%p17, %rs11, %rs1;
	cvt.u32.u16	%r140, %rs11;
	selp.b32	%r143, %r94, %r140, %p17;
	selp.b32	%r144, %r140, %r94, %p17;
	add.s32 	%r145, %r144, 1;
	mul.lo.s32 	%r146, %r145, %r144;
	shr.u32 	%r147, %r146, 1;
	add.s32 	%r148, %r147, %r143;
	mul.wide.s32 	%rd60, %r148, 4;
	add.s64 	%rd61, %rd33, %rd60;
	ld.global.f32 	%f75, [%rd61];
	mul.f32 	%f76, %f75, %f31;
	sub.f32 	%f77, %f72, %f1;
	sub.f32 	%f78, %f73, %f2;
	sub.f32 	%f79, %f74, %f3;
	fma.rn.f32 	%f14, %f76, %f77, %f11;
	fma.rn.f32 	%f15, %f76, %f78, %f12;
	fma.rn.f32 	%f16, %f76, %f79, %f13;
	add.s32 	%r15, %r2, 1;
	@%p15 bra 	BB0_17;

	rem.s32 	%r153, %r15, %r28;
	add.s32 	%r154, %r153, %r28;
	rem.s32 	%r244, %r154, %r28;
	bra.uni 	BB0_18;

BB0_17:
	add.s32 	%r155, %r28, -1;
	min.s32 	%r244, %r15, %r155;

BB0_18:
	mad.lo.s32 	%r160, %r3, %r28, %r244;
	mad.lo.s32 	%r165, %r160, %r27, %r1;
	cvt.s64.s32	%rd63, %r165;
	mul.wide.s32 	%rd64, %r165, 4;
	add.s64 	%rd65, %rd10, %rd64;
	add.s64 	%rd67, %rd13, %rd64;
	add.s64 	%rd69, %rd15, %rd64;
	ld.global.f32 	%f80, [%rd65];
	ld.global.f32 	%f81, [%rd67];
	mul.f32 	%f82, %f81, %f81;
	fma.rn.f32 	%f83, %f80, %f80, %f82;
	ld.global.f32 	%f84, [%rd69];
	fma.rn.f32 	%f85, %f84, %f84, %f83;
	setp.eq.f32	%p19, %f85, 0f00000000;
	selp.f32	%f86, %f1, %f80, %p19;
	selp.f32	%f87, %f2, %f81, %p19;
	selp.f32	%f88, %f3, %f84, %p19;
	add.s64 	%rd71, %rd20, %rd63;
	ld.global.u8 	%rs14, [%rd71];
	setp.gt.u16	%p20, %rs14, %rs1;
	cvt.u32.u16	%r166, %rs14;
	selp.b32	%r169, %r94, %r166, %p20;
	selp.b32	%r170, %r166, %r94, %p20;
	add.s32 	%r171, %r170, 1;
	mul.lo.s32 	%r172, %r171, %r170;
	shr.u32 	%r173, %r172, 1;
	add.s32 	%r174, %r173, %r169;
	mul.wide.s32 	%rd73, %r174, 4;
	add.s64 	%rd74, %rd33, %rd73;
	ld.global.f32 	%f89, [%rd74];
	mul.f32 	%f90, %f89, %f31;
	sub.f32 	%f91, %f86, %f1;
	sub.f32 	%f92, %f87, %f2;
	sub.f32 	%f93, %f88, %f3;
	fma.rn.f32 	%f130, %f90, %f91, %f14;
	fma.rn.f32 	%f131, %f90, %f92, %f15;
	fma.rn.f32 	%f132, %f90, %f93, %f16;
	setp.eq.s32	%p21, %r29, 1;
	@%p21 bra 	BB0_26;

	and.b16  	%rs3, %rs4, 4;
	setp.eq.s16	%p22, %rs3, 0;
	add.s32 	%r19, %r3, -1;
	@%p22 bra 	BB0_21;

	rem.s32 	%r179, %r19, %r29;
	add.s32 	%r180, %r179, %r29;
	rem.s32 	%r245, %r180, %r29;
	bra.uni 	BB0_22;

BB0_21:
	mov.u32 	%r181, 0;
	max.s32 	%r245, %r19, %r181;

BB0_22:
	mad.lo.s32 	%r186, %r245, %r28, %r2;
	mad.lo.s32 	%r191, %r186, %r27, %r1;
	cvt.s64.s32	%rd76, %r191;
	mul.wide.s32 	%rd77, %r191, 4;
	add.s64 	%rd78, %rd10, %rd77;
	add.s64 	%rd80, %rd13, %rd77;
	add.s64 	%rd82, %rd15, %rd77;
	ld.global.f32 	%f94, [%rd78];
	ld.global.f32 	%f95, [%rd80];
	mul.f32 	%f96, %f95, %f95;
	fma.rn.f32 	%f97, %f94, %f94, %f96;
	ld.global.f32 	%f98, [%rd82];
	fma.rn.f32 	%f99, %f98, %f98, %f97;
	setp.eq.f32	%p23, %f99, 0f00000000;
	selp.f32	%f100, %f1, %f94, %p23;
	selp.f32	%f101, %f2, %f95, %p23;
	selp.f32	%f102, %f3, %f98, %p23;
	add.s64 	%rd84, %rd20, %rd76;
	ld.global.u8 	%rs16, [%rd84];
	setp.gt.u16	%p24, %rs16, %rs1;
	cvt.u32.u16	%r192, %rs16;
	selp.b32	%r195, %r94, %r192, %p24;
	selp.b32	%r196, %r192, %r94, %p24;
	add.s32 	%r197, %r196, 1;
	mul.lo.s32 	%r198, %r197, %r196;
	shr.u32 	%r199, %r198, 1;
	add.s32 	%r200, %r199, %r195;
	mul.wide.s32 	%rd86, %r200, 4;
	add.s64 	%rd87, %rd33, %rd86;
	ld.global.f32 	%f103, [%rd87];
	mul.f32 	%f104, %f103, %f32;
	sub.f32 	%f105, %f100, %f1;
	sub.f32 	%f106, %f101, %f2;
	sub.f32 	%f107, %f102, %f3;
	fma.rn.f32 	%f20, %f104, %f105, %f130;
	fma.rn.f32 	%f21, %f104, %f106, %f131;
	fma.rn.f32 	%f22, %f104, %f107, %f132;
	add.s32 	%r23, %r3, 1;
	@%p22 bra 	BB0_24;

	rem.s32 	%r205, %r23, %r29;
	add.s32 	%r206, %r205, %r29;
	rem.s32 	%r246, %r206, %r29;
	bra.uni 	BB0_25;

BB0_24:
	add.s32 	%r207, %r29, -1;
	min.s32 	%r246, %r23, %r207;

BB0_25:
	mad.lo.s32 	%r212, %r246, %r28, %r2;
	mad.lo.s32 	%r217, %r212, %r27, %r1;
	cvt.s64.s32	%rd89, %r217;
	mul.wide.s32 	%rd90, %r217, 4;
	add.s64 	%rd91, %rd10, %rd90;
	add.s64 	%rd93, %rd13, %rd90;
	add.s64 	%rd95, %rd15, %rd90;
	ld.global.f32 	%f108, [%rd91];
	ld.global.f32 	%f109, [%rd93];
	mul.f32 	%f110, %f109, %f109;
	fma.rn.f32 	%f111, %f108, %f108, %f110;
	ld.global.f32 	%f112, [%rd95];
	fma.rn.f32 	%f113, %f112, %f112, %f111;
	setp.eq.f32	%p26, %f113, 0f00000000;
	selp.f32	%f114, %f1, %f108, %p26;
	selp.f32	%f115, %f2, %f109, %p26;
	selp.f32	%f116, %f3, %f112, %p26;
	add.s64 	%rd97, %rd20, %rd89;
	ld.global.u8 	%rs19, [%rd97];
	setp.gt.u16	%p27, %rs19, %rs1;
	cvt.u32.u16	%r218, %rs19;
	selp.b32	%r221, %r94, %r218, %p27;
	selp.b32	%r222, %r218, %r94, %p27;
	add.s32 	%r223, %r222, 1;
	mul.lo.s32 	%r224, %r223, %r222;
	shr.u32 	%r225, %r224, 1;
	add.s32 	%r226, %r225, %r221;
	mul.wide.s32 	%rd99, %r226, 4;
	add.s64 	%rd100, %rd33, %rd99;
	ld.global.f32 	%f117, [%rd100];
	mul.f32 	%f118, %f117, %f32;
	sub.f32 	%f119, %f114, %f1;
	sub.f32 	%f120, %f115, %f2;
	sub.f32 	%f121, %f116, %f3;
	fma.rn.f32 	%f130, %f118, %f119, %f20;
	fma.rn.f32 	%f131, %f118, %f120, %f21;
	fma.rn.f32 	%f132, %f118, %f121, %f22;

BB0_26:
	cvta.to.global.u64 	%rd101, %rd1;
	add.s64 	%rd103, %rd101, %rd11;
	ld.global.f32 	%f122, [%rd103];
	fma.rn.f32 	%f123, %f129, %f130, %f122;
	st.global.f32 	[%rd103], %f123;
	cvta.to.global.u64 	%rd104, %rd2;
	add.s64 	%rd105, %rd104, %rd11;
	ld.global.f32 	%f124, [%rd105];
	fma.rn.f32 	%f125, %f129, %f131, %f124;
	st.global.f32 	[%rd105], %f125;
	cvta.to.global.u64 	%rd106, %rd3;
	add.s64 	%rd107, %rd106, %rd11;
	ld.global.f32 	%f126, [%rd107];
	fma.rn.f32 	%f127, %f129, %f132, %f126;
	st.global.f32 	[%rd107], %f127;

BB0_27:
	ret;
}


`
)
