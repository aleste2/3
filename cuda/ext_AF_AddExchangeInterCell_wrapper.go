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
	70: addexchangeAfll_ptx_70}

// addexchangeAfll PTX code for various compute capabilities.
const (
	addexchangeAfll_ptx_70 = `
.version 7.1
.target sm_70
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
	.reg .b16 	%rs<26>;
	.reg .f32 	%f<133>;
	.reg .b32 	%r<128>;
	.reg .b64 	%rd<79>;


	ld.param.u64 	%rd6, [addexchangeAfll_param_0];
	ld.param.u64 	%rd7, [addexchangeAfll_param_1];
	ld.param.u64 	%rd8, [addexchangeAfll_param_2];
	ld.param.u64 	%rd10, [addexchangeAfll_param_9];
	ld.param.u64 	%rd11, [addexchangeAfll_param_10];
	ld.param.u64 	%rd12, [addexchangeAfll_param_11];
	ld.param.u64 	%rd9, [addexchangeAfll_param_12];
	ld.param.f32 	%f128, [addexchangeAfll_param_13];
	ld.param.u64 	%rd13, [addexchangeAfll_param_16];
	ld.param.u64 	%rd14, [addexchangeAfll_param_17];
	ld.param.f32 	%f30, [addexchangeAfll_param_18];
	ld.param.f32 	%f31, [addexchangeAfll_param_19];
	ld.param.f32 	%f32, [addexchangeAfll_param_20];
	ld.param.u32 	%r32, [addexchangeAfll_param_21];
	ld.param.u32 	%r33, [addexchangeAfll_param_22];
	ld.param.u32 	%r34, [addexchangeAfll_param_23];
	ld.param.u8 	%rs5, [addexchangeAfll_param_24];
	cvta.to.global.u64 	%rd1, %rd13;
	cvta.to.global.u64 	%rd2, %rd14;
	cvta.to.global.u64 	%rd3, %rd12;
	cvta.to.global.u64 	%rd4, %rd11;
	cvta.to.global.u64 	%rd5, %rd10;
	mov.u32 	%r35, %ntid.x;
	mov.u32 	%r36, %ctaid.x;
	mov.u32 	%r37, %tid.x;
	mad.lo.s32 	%r1, %r35, %r36, %r37;
	mov.u32 	%r38, %ntid.y;
	mov.u32 	%r39, %ctaid.y;
	mov.u32 	%r40, %tid.y;
	mad.lo.s32 	%r2, %r38, %r39, %r40;
	mov.u32 	%r41, %ntid.z;
	mov.u32 	%r42, %ctaid.z;
	mov.u32 	%r43, %tid.z;
	mad.lo.s32 	%r3, %r41, %r42, %r43;
	setp.ge.s32	%p1, %r2, %r33;
	setp.ge.s32	%p2, %r1, %r32;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r3, %r34;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_27;

	mul.lo.s32 	%r4, %r3, %r33;
	add.s32 	%r44, %r4, %r2;
	mul.lo.s32 	%r5, %r44, %r32;
	add.s32 	%r6, %r5, %r1;
	mul.wide.s32 	%rd15, %r6, 4;
	add.s64 	%rd16, %rd5, %rd15;
	ld.global.nc.f32 	%f1, [%rd16];
	add.s64 	%rd17, %rd4, %rd15;
	ld.global.nc.f32 	%f2, [%rd17];
	add.s64 	%rd18, %rd3, %rd15;
	ld.global.nc.f32 	%f3, [%rd18];
	setp.eq.s64	%p6, %rd9, 0;
	@%p6 bra 	BB0_3;

	cvta.to.global.u64 	%rd19, %rd9;
	add.s64 	%rd21, %rd19, %rd15;
	ld.global.nc.f32 	%f33, [%rd21];
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

	cvt.s64.s32	%rd22, %r6;
	add.s64 	%rd23, %rd2, %rd22;
	ld.global.nc.u8 	%rs1, [%rd23];
	cvt.u32.u16	%r45, %rs1;
	and.b32  	%r7, %r45, 255;
	and.b16  	%rs2, %rs5, 1;
	setp.eq.s16	%p9, %rs2, 0;
	add.s32 	%r8, %r1, -1;
	@%p9 bra 	BB0_8;

	rem.s32 	%r46, %r8, %r32;
	add.s32 	%r47, %r46, %r32;
	rem.s32 	%r122, %r47, %r32;
	bra.uni 	BB0_9;

BB0_8:
	mov.u32 	%r48, 0;
	max.s32 	%r122, %r8, %r48;

BB0_9:
	add.s32 	%r49, %r122, %r5;
	cvt.s64.s32	%rd24, %r49;
	mul.wide.s32 	%rd25, %r49, 4;
	add.s64 	%rd26, %rd5, %rd25;
	add.s64 	%rd27, %rd4, %rd25;
	add.s64 	%rd28, %rd3, %rd25;
	ld.global.nc.f32 	%f38, [%rd26];
	ld.global.nc.f32 	%f39, [%rd27];
	mul.f32 	%f40, %f39, %f39;
	fma.rn.f32 	%f41, %f38, %f38, %f40;
	ld.global.nc.f32 	%f42, [%rd28];
	fma.rn.f32 	%f43, %f42, %f42, %f41;
	setp.eq.f32	%p10, %f43, 0f00000000;
	selp.f32	%f44, %f1, %f38, %p10;
	selp.f32	%f45, %f2, %f39, %p10;
	selp.f32	%f46, %f3, %f42, %p10;
	add.s64 	%rd29, %rd2, %rd24;
	ld.global.nc.u8 	%rs6, [%rd29];
	setp.gt.u16	%p11, %rs6, %rs1;
	cvt.u32.u16	%r50, %rs6;
	and.b32  	%r51, %r50, 255;
	selp.b32	%r52, %r7, %r51, %p11;
	selp.b32	%r53, %r51, %r7, %p11;
	add.s32 	%r54, %r53, 1;
	mul.lo.s32 	%r55, %r54, %r53;
	shr.u32 	%r56, %r55, 1;
	add.s32 	%r57, %r56, %r52;
	mul.wide.s32 	%rd30, %r57, 4;
	add.s64 	%rd31, %rd1, %rd30;
	ld.global.nc.f32 	%f47, [%rd31];
	mul.f32 	%f48, %f47, %f30;
	sub.f32 	%f49, %f44, %f1;
	sub.f32 	%f50, %f45, %f2;
	sub.f32 	%f51, %f46, %f3;
	fma.rn.f32 	%f8, %f48, %f49, 0f00000000;
	fma.rn.f32 	%f9, %f48, %f50, 0f00000000;
	fma.rn.f32 	%f10, %f48, %f51, 0f00000000;
	add.s32 	%r12, %r1, 1;
	@%p9 bra 	BB0_11;

	rem.s32 	%r58, %r12, %r32;
	add.s32 	%r59, %r58, %r32;
	rem.s32 	%r123, %r59, %r32;
	bra.uni 	BB0_12;

BB0_11:
	add.s32 	%r60, %r32, -1;
	min.s32 	%r123, %r12, %r60;

BB0_12:
	add.s32 	%r61, %r123, %r5;
	cvt.s64.s32	%rd32, %r61;
	mul.wide.s32 	%rd33, %r61, 4;
	add.s64 	%rd34, %rd5, %rd33;
	add.s64 	%rd35, %rd4, %rd33;
	add.s64 	%rd36, %rd3, %rd33;
	ld.global.nc.f32 	%f52, [%rd34];
	ld.global.nc.f32 	%f53, [%rd35];
	mul.f32 	%f54, %f53, %f53;
	fma.rn.f32 	%f55, %f52, %f52, %f54;
	ld.global.nc.f32 	%f56, [%rd36];
	fma.rn.f32 	%f57, %f56, %f56, %f55;
	setp.eq.f32	%p13, %f57, 0f00000000;
	selp.f32	%f58, %f1, %f52, %p13;
	selp.f32	%f59, %f2, %f53, %p13;
	selp.f32	%f60, %f3, %f56, %p13;
	add.s64 	%rd37, %rd2, %rd32;
	ld.global.nc.u8 	%rs9, [%rd37];
	setp.gt.u16	%p14, %rs9, %rs1;
	cvt.u32.u16	%r62, %rs9;
	and.b32  	%r63, %r62, 255;
	selp.b32	%r64, %r7, %r63, %p14;
	selp.b32	%r65, %r63, %r7, %p14;
	add.s32 	%r66, %r65, 1;
	mul.lo.s32 	%r67, %r66, %r65;
	shr.u32 	%r68, %r67, 1;
	add.s32 	%r69, %r68, %r64;
	mul.wide.s32 	%rd38, %r69, 4;
	add.s64 	%rd39, %rd1, %rd38;
	ld.global.nc.f32 	%f61, [%rd39];
	mul.f32 	%f62, %f61, %f30;
	sub.f32 	%f63, %f58, %f1;
	sub.f32 	%f64, %f59, %f2;
	sub.f32 	%f65, %f60, %f3;
	fma.rn.f32 	%f11, %f62, %f63, %f8;
	fma.rn.f32 	%f12, %f62, %f64, %f9;
	fma.rn.f32 	%f13, %f62, %f65, %f10;
	and.b16  	%rs3, %rs5, 2;
	setp.eq.s16	%p15, %rs3, 0;
	add.s32 	%r16, %r2, -1;
	@%p15 bra 	BB0_14;

	rem.s32 	%r70, %r16, %r33;
	add.s32 	%r71, %r70, %r33;
	rem.s32 	%r124, %r71, %r33;
	bra.uni 	BB0_15;

BB0_14:
	mov.u32 	%r72, 0;
	max.s32 	%r124, %r16, %r72;

BB0_15:
	add.s32 	%r73, %r124, %r4;
	mad.lo.s32 	%r74, %r73, %r32, %r1;
	cvt.s64.s32	%rd40, %r74;
	mul.wide.s32 	%rd41, %r74, 4;
	add.s64 	%rd42, %rd5, %rd41;
	add.s64 	%rd43, %rd4, %rd41;
	add.s64 	%rd44, %rd3, %rd41;
	ld.global.nc.f32 	%f66, [%rd42];
	ld.global.nc.f32 	%f67, [%rd43];
	mul.f32 	%f68, %f67, %f67;
	fma.rn.f32 	%f69, %f66, %f66, %f68;
	ld.global.nc.f32 	%f70, [%rd44];
	fma.rn.f32 	%f71, %f70, %f70, %f69;
	setp.eq.f32	%p16, %f71, 0f00000000;
	selp.f32	%f72, %f1, %f66, %p16;
	selp.f32	%f73, %f2, %f67, %p16;
	selp.f32	%f74, %f3, %f70, %p16;
	add.s64 	%rd45, %rd2, %rd40;
	ld.global.nc.u8 	%rs12, [%rd45];
	setp.gt.u16	%p17, %rs12, %rs1;
	cvt.u32.u16	%r75, %rs12;
	and.b32  	%r76, %r75, 255;
	selp.b32	%r77, %r7, %r76, %p17;
	selp.b32	%r78, %r76, %r7, %p17;
	add.s32 	%r79, %r78, 1;
	mul.lo.s32 	%r80, %r79, %r78;
	shr.u32 	%r81, %r80, 1;
	add.s32 	%r82, %r81, %r77;
	mul.wide.s32 	%rd46, %r82, 4;
	add.s64 	%rd47, %rd1, %rd46;
	ld.global.nc.f32 	%f75, [%rd47];
	mul.f32 	%f76, %f75, %f31;
	sub.f32 	%f77, %f72, %f1;
	sub.f32 	%f78, %f73, %f2;
	sub.f32 	%f79, %f74, %f3;
	fma.rn.f32 	%f14, %f76, %f77, %f11;
	fma.rn.f32 	%f15, %f76, %f78, %f12;
	fma.rn.f32 	%f16, %f76, %f79, %f13;
	add.s32 	%r20, %r2, 1;
	@%p15 bra 	BB0_17;

	rem.s32 	%r83, %r20, %r33;
	add.s32 	%r84, %r83, %r33;
	rem.s32 	%r125, %r84, %r33;
	bra.uni 	BB0_18;

BB0_17:
	add.s32 	%r85, %r33, -1;
	min.s32 	%r125, %r20, %r85;

BB0_18:
	add.s32 	%r86, %r125, %r4;
	mad.lo.s32 	%r87, %r86, %r32, %r1;
	cvt.s64.s32	%rd48, %r87;
	mul.wide.s32 	%rd49, %r87, 4;
	add.s64 	%rd50, %rd5, %rd49;
	add.s64 	%rd51, %rd4, %rd49;
	add.s64 	%rd52, %rd3, %rd49;
	ld.global.nc.f32 	%f80, [%rd50];
	ld.global.nc.f32 	%f81, [%rd51];
	mul.f32 	%f82, %f81, %f81;
	fma.rn.f32 	%f83, %f80, %f80, %f82;
	ld.global.nc.f32 	%f84, [%rd52];
	fma.rn.f32 	%f85, %f84, %f84, %f83;
	setp.eq.f32	%p19, %f85, 0f00000000;
	selp.f32	%f86, %f1, %f80, %p19;
	selp.f32	%f87, %f2, %f81, %p19;
	selp.f32	%f88, %f3, %f84, %p19;
	add.s64 	%rd53, %rd2, %rd48;
	ld.global.nc.u8 	%rs16, [%rd53];
	setp.gt.u16	%p20, %rs16, %rs1;
	cvt.u32.u16	%r88, %rs16;
	and.b32  	%r89, %r88, 255;
	selp.b32	%r90, %r7, %r89, %p20;
	selp.b32	%r91, %r89, %r7, %p20;
	add.s32 	%r92, %r91, 1;
	mul.lo.s32 	%r93, %r92, %r91;
	shr.u32 	%r94, %r93, 1;
	add.s32 	%r95, %r94, %r90;
	mul.wide.s32 	%rd54, %r95, 4;
	add.s64 	%rd55, %rd1, %rd54;
	ld.global.nc.f32 	%f89, [%rd55];
	mul.f32 	%f90, %f89, %f31;
	sub.f32 	%f91, %f86, %f1;
	sub.f32 	%f92, %f87, %f2;
	sub.f32 	%f93, %f88, %f3;
	fma.rn.f32 	%f130, %f90, %f91, %f14;
	fma.rn.f32 	%f131, %f90, %f92, %f15;
	fma.rn.f32 	%f132, %f90, %f93, %f16;
	setp.eq.s32	%p21, %r34, 1;
	@%p21 bra 	BB0_26;

	and.b16  	%rs4, %rs5, 4;
	setp.eq.s16	%p22, %rs4, 0;
	add.s32 	%r24, %r3, -1;
	@%p22 bra 	BB0_21;

	rem.s32 	%r96, %r24, %r34;
	add.s32 	%r97, %r96, %r34;
	rem.s32 	%r126, %r97, %r34;
	bra.uni 	BB0_22;

BB0_21:
	mov.u32 	%r98, 0;
	max.s32 	%r126, %r24, %r98;

BB0_22:
	mad.lo.s32 	%r99, %r126, %r33, %r2;
	mad.lo.s32 	%r100, %r99, %r32, %r1;
	cvt.s64.s32	%rd56, %r100;
	mul.wide.s32 	%rd57, %r100, 4;
	add.s64 	%rd58, %rd5, %rd57;
	add.s64 	%rd59, %rd4, %rd57;
	add.s64 	%rd60, %rd3, %rd57;
	ld.global.nc.f32 	%f94, [%rd58];
	ld.global.nc.f32 	%f95, [%rd59];
	mul.f32 	%f96, %f95, %f95;
	fma.rn.f32 	%f97, %f94, %f94, %f96;
	ld.global.nc.f32 	%f98, [%rd60];
	fma.rn.f32 	%f99, %f98, %f98, %f97;
	setp.eq.f32	%p23, %f99, 0f00000000;
	selp.f32	%f100, %f1, %f94, %p23;
	selp.f32	%f101, %f2, %f95, %p23;
	selp.f32	%f102, %f3, %f98, %p23;
	add.s64 	%rd61, %rd2, %rd56;
	ld.global.nc.u8 	%rs19, [%rd61];
	setp.gt.u16	%p24, %rs19, %rs1;
	cvt.u32.u16	%r101, %rs19;
	and.b32  	%r102, %r101, 255;
	selp.b32	%r103, %r7, %r102, %p24;
	selp.b32	%r104, %r102, %r7, %p24;
	add.s32 	%r105, %r104, 1;
	mul.lo.s32 	%r106, %r105, %r104;
	shr.u32 	%r107, %r106, 1;
	add.s32 	%r108, %r107, %r103;
	mul.wide.s32 	%rd62, %r108, 4;
	add.s64 	%rd63, %rd1, %rd62;
	ld.global.nc.f32 	%f103, [%rd63];
	mul.f32 	%f104, %f103, %f32;
	sub.f32 	%f105, %f100, %f1;
	sub.f32 	%f106, %f101, %f2;
	sub.f32 	%f107, %f102, %f3;
	fma.rn.f32 	%f20, %f104, %f105, %f130;
	fma.rn.f32 	%f21, %f104, %f106, %f131;
	fma.rn.f32 	%f22, %f104, %f107, %f132;
	add.s32 	%r28, %r3, 1;
	@%p22 bra 	BB0_24;

	rem.s32 	%r109, %r28, %r34;
	add.s32 	%r110, %r109, %r34;
	rem.s32 	%r127, %r110, %r34;
	bra.uni 	BB0_25;

BB0_24:
	add.s32 	%r111, %r34, -1;
	min.s32 	%r127, %r28, %r111;

BB0_25:
	mad.lo.s32 	%r112, %r127, %r33, %r2;
	mad.lo.s32 	%r113, %r112, %r32, %r1;
	cvt.s64.s32	%rd64, %r113;
	mul.wide.s32 	%rd65, %r113, 4;
	add.s64 	%rd66, %rd5, %rd65;
	add.s64 	%rd67, %rd4, %rd65;
	add.s64 	%rd68, %rd3, %rd65;
	ld.global.nc.f32 	%f108, [%rd66];
	ld.global.nc.f32 	%f109, [%rd67];
	mul.f32 	%f110, %f109, %f109;
	fma.rn.f32 	%f111, %f108, %f108, %f110;
	ld.global.nc.f32 	%f112, [%rd68];
	fma.rn.f32 	%f113, %f112, %f112, %f111;
	setp.eq.f32	%p26, %f113, 0f00000000;
	selp.f32	%f114, %f1, %f108, %p26;
	selp.f32	%f115, %f2, %f109, %p26;
	selp.f32	%f116, %f3, %f112, %p26;
	add.s64 	%rd69, %rd2, %rd64;
	ld.global.nc.u8 	%rs23, [%rd69];
	setp.gt.u16	%p27, %rs23, %rs1;
	cvt.u32.u16	%r114, %rs23;
	and.b32  	%r115, %r114, 255;
	selp.b32	%r116, %r7, %r115, %p27;
	selp.b32	%r117, %r115, %r7, %p27;
	add.s32 	%r118, %r117, 1;
	mul.lo.s32 	%r119, %r118, %r117;
	shr.u32 	%r120, %r119, 1;
	add.s32 	%r121, %r120, %r116;
	mul.wide.s32 	%rd70, %r121, 4;
	add.s64 	%rd71, %rd1, %rd70;
	ld.global.nc.f32 	%f117, [%rd71];
	mul.f32 	%f118, %f117, %f32;
	sub.f32 	%f119, %f114, %f1;
	sub.f32 	%f120, %f115, %f2;
	sub.f32 	%f121, %f116, %f3;
	fma.rn.f32 	%f130, %f118, %f119, %f20;
	fma.rn.f32 	%f131, %f118, %f120, %f21;
	fma.rn.f32 	%f132, %f118, %f121, %f22;

BB0_26:
	cvta.to.global.u64 	%rd72, %rd6;
	add.s64 	%rd74, %rd72, %rd15;
	ld.global.f32 	%f122, [%rd74];
	fma.rn.f32 	%f123, %f129, %f130, %f122;
	st.global.f32 	[%rd74], %f123;
	cvta.to.global.u64 	%rd75, %rd7;
	add.s64 	%rd76, %rd75, %rd15;
	ld.global.f32 	%f124, [%rd76];
	fma.rn.f32 	%f125, %f129, %f131, %f124;
	st.global.f32 	[%rd76], %f125;
	cvta.to.global.u64 	%rd77, %rd8;
	add.s64 	%rd78, %rd77, %rd15;
	ld.global.f32 	%f126, [%rd78];
	fma.rn.f32 	%f127, %f129, %f132, %f126;
	st.global.f32 	[%rd78], %f127;

BB0_27:
	ret;
}


`
)
