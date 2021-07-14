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

// CUDA handle for addexchange kernel
var addexchange_code cu.Function

// Stores the arguments for addexchange kernel invocation
type addexchange_args_t struct {
	arg_Bx      unsafe.Pointer
	arg_By      unsafe.Pointer
	arg_Bz      unsafe.Pointer
	arg_mx      unsafe.Pointer
	arg_my      unsafe.Pointer
	arg_mz      unsafe.Pointer
	arg_Ms_     unsafe.Pointer
	arg_Ms_mul  float32
	arg_aLUT2d  unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_wx      float32
	arg_wy      float32
	arg_wz      float32
	arg_Nx      int
	arg_Ny      int
	arg_Nz      int
	arg_PBC     byte
	argptr      [17]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for addexchange kernel invocation
var addexchange_args addexchange_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	addexchange_args.argptr[0] = unsafe.Pointer(&addexchange_args.arg_Bx)
	addexchange_args.argptr[1] = unsafe.Pointer(&addexchange_args.arg_By)
	addexchange_args.argptr[2] = unsafe.Pointer(&addexchange_args.arg_Bz)
	addexchange_args.argptr[3] = unsafe.Pointer(&addexchange_args.arg_mx)
	addexchange_args.argptr[4] = unsafe.Pointer(&addexchange_args.arg_my)
	addexchange_args.argptr[5] = unsafe.Pointer(&addexchange_args.arg_mz)
	addexchange_args.argptr[6] = unsafe.Pointer(&addexchange_args.arg_Ms_)
	addexchange_args.argptr[7] = unsafe.Pointer(&addexchange_args.arg_Ms_mul)
	addexchange_args.argptr[8] = unsafe.Pointer(&addexchange_args.arg_aLUT2d)
	addexchange_args.argptr[9] = unsafe.Pointer(&addexchange_args.arg_regions)
	addexchange_args.argptr[10] = unsafe.Pointer(&addexchange_args.arg_wx)
	addexchange_args.argptr[11] = unsafe.Pointer(&addexchange_args.arg_wy)
	addexchange_args.argptr[12] = unsafe.Pointer(&addexchange_args.arg_wz)
	addexchange_args.argptr[13] = unsafe.Pointer(&addexchange_args.arg_Nx)
	addexchange_args.argptr[14] = unsafe.Pointer(&addexchange_args.arg_Ny)
	addexchange_args.argptr[15] = unsafe.Pointer(&addexchange_args.arg_Nz)
	addexchange_args.argptr[16] = unsafe.Pointer(&addexchange_args.arg_PBC)
}

// Wrapper for addexchange CUDA kernel, asynchronous.
func k_addexchange_async(Bx unsafe.Pointer, By unsafe.Pointer, Bz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Ms_ unsafe.Pointer, Ms_mul float32, aLUT2d unsafe.Pointer, regions unsafe.Pointer, wx float32, wy float32, wz float32, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("addexchange")
	}

	addexchange_args.Lock()
	defer addexchange_args.Unlock()

	if addexchange_code == 0 {
		addexchange_code = fatbinLoad(addexchange_map, "addexchange")
	}

	addexchange_args.arg_Bx = Bx
	addexchange_args.arg_By = By
	addexchange_args.arg_Bz = Bz
	addexchange_args.arg_mx = mx
	addexchange_args.arg_my = my
	addexchange_args.arg_mz = mz
	addexchange_args.arg_Ms_ = Ms_
	addexchange_args.arg_Ms_mul = Ms_mul
	addexchange_args.arg_aLUT2d = aLUT2d
	addexchange_args.arg_regions = regions
	addexchange_args.arg_wx = wx
	addexchange_args.arg_wy = wy
	addexchange_args.arg_wz = wz
	addexchange_args.arg_Nx = Nx
	addexchange_args.arg_Ny = Ny
	addexchange_args.arg_Nz = Nz
	addexchange_args.arg_PBC = PBC

	args := addexchange_args.argptr[:]
	cu.LaunchKernel(addexchange_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("addexchange")
	}
}

// maps compute capability on PTX code for addexchange kernel.
var addexchange_map = map[int]string{0: "",
	70: addexchange_ptx_70}

// addexchange PTX code for various compute capabilities.
const (
	addexchange_ptx_70 = `
.version 7.1
.target sm_70
.address_size 64

	// .globl	addexchange

.visible .entry addexchange(
	.param .u64 addexchange_param_0,
	.param .u64 addexchange_param_1,
	.param .u64 addexchange_param_2,
	.param .u64 addexchange_param_3,
	.param .u64 addexchange_param_4,
	.param .u64 addexchange_param_5,
	.param .u64 addexchange_param_6,
	.param .f32 addexchange_param_7,
	.param .u64 addexchange_param_8,
	.param .u64 addexchange_param_9,
	.param .f32 addexchange_param_10,
	.param .f32 addexchange_param_11,
	.param .f32 addexchange_param_12,
	.param .u32 addexchange_param_13,
	.param .u32 addexchange_param_14,
	.param .u32 addexchange_param_15,
	.param .u8 addexchange_param_16
)
{
	.reg .pred 	%p<28>;
	.reg .b16 	%rs<26>;
	.reg .f32 	%f<133>;
	.reg .b32 	%r<128>;
	.reg .b64 	%rd<79>;


	ld.param.u64 	%rd6, [addexchange_param_0];
	ld.param.u64 	%rd7, [addexchange_param_1];
	ld.param.u64 	%rd8, [addexchange_param_2];
	ld.param.u64 	%rd10, [addexchange_param_3];
	ld.param.u64 	%rd11, [addexchange_param_4];
	ld.param.u64 	%rd12, [addexchange_param_5];
	ld.param.u64 	%rd9, [addexchange_param_6];
	ld.param.f32 	%f131, [addexchange_param_7];
	ld.param.u64 	%rd13, [addexchange_param_8];
	ld.param.u64 	%rd14, [addexchange_param_9];
	ld.param.f32 	%f30, [addexchange_param_10];
	ld.param.f32 	%f31, [addexchange_param_11];
	ld.param.f32 	%f32, [addexchange_param_12];
	ld.param.u32 	%r32, [addexchange_param_13];
	ld.param.u32 	%r33, [addexchange_param_14];
	ld.param.u32 	%r34, [addexchange_param_15];
	ld.param.u8 	%rs5, [addexchange_param_16];
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
	add.s64 	%rd17, %rd4, %rd15;
	add.s64 	%rd18, %rd3, %rd15;
	ld.global.nc.f32 	%f1, [%rd16];
	ld.global.nc.f32 	%f2, [%rd17];
	mul.f32 	%f33, %f2, %f2;
	fma.rn.f32 	%f34, %f1, %f1, %f33;
	ld.global.nc.f32 	%f3, [%rd18];
	fma.rn.f32 	%f35, %f3, %f3, %f34;
	setp.eq.f32	%p6, %f35, 0f00000000;
	@%p6 bra 	BB0_27;

	cvt.s64.s32	%rd19, %r6;
	add.s64 	%rd20, %rd2, %rd19;
	ld.global.nc.u8 	%rs1, [%rd20];
	cvt.u32.u16	%r45, %rs1;
	and.b32  	%r7, %r45, 255;
	and.b16  	%rs2, %rs5, 1;
	setp.eq.s16	%p7, %rs2, 0;
	add.s32 	%r8, %r1, -1;
	@%p7 bra 	BB0_4;

	rem.s32 	%r46, %r8, %r32;
	add.s32 	%r47, %r46, %r32;
	rem.s32 	%r122, %r47, %r32;
	bra.uni 	BB0_5;

BB0_4:
	mov.u32 	%r48, 0;
	max.s32 	%r122, %r8, %r48;

BB0_5:
	add.s32 	%r49, %r122, %r5;
	cvt.s64.s32	%rd21, %r49;
	mul.wide.s32 	%rd22, %r49, 4;
	add.s64 	%rd23, %rd5, %rd22;
	add.s64 	%rd24, %rd4, %rd22;
	add.s64 	%rd25, %rd3, %rd22;
	ld.global.nc.f32 	%f36, [%rd23];
	ld.global.nc.f32 	%f37, [%rd24];
	mul.f32 	%f38, %f37, %f37;
	fma.rn.f32 	%f39, %f36, %f36, %f38;
	ld.global.nc.f32 	%f40, [%rd25];
	fma.rn.f32 	%f41, %f40, %f40, %f39;
	setp.eq.f32	%p8, %f41, 0f00000000;
	selp.f32	%f42, %f1, %f36, %p8;
	selp.f32	%f43, %f2, %f37, %p8;
	selp.f32	%f44, %f3, %f40, %p8;
	add.s64 	%rd26, %rd2, %rd21;
	ld.global.nc.u8 	%rs6, [%rd26];
	setp.gt.u16	%p9, %rs6, %rs1;
	cvt.u32.u16	%r50, %rs6;
	and.b32  	%r51, %r50, 255;
	selp.b32	%r52, %r7, %r51, %p9;
	selp.b32	%r53, %r51, %r7, %p9;
	add.s32 	%r54, %r53, 1;
	mul.lo.s32 	%r55, %r54, %r53;
	shr.u32 	%r56, %r55, 1;
	add.s32 	%r57, %r56, %r52;
	mul.wide.s32 	%rd27, %r57, 4;
	add.s64 	%rd28, %rd1, %rd27;
	ld.global.nc.f32 	%f45, [%rd28];
	mul.f32 	%f46, %f45, %f30;
	sub.f32 	%f47, %f42, %f1;
	sub.f32 	%f48, %f43, %f2;
	sub.f32 	%f49, %f44, %f3;
	fma.rn.f32 	%f4, %f46, %f47, 0f00000000;
	fma.rn.f32 	%f5, %f46, %f48, 0f00000000;
	fma.rn.f32 	%f6, %f46, %f49, 0f00000000;
	add.s32 	%r12, %r1, 1;
	@%p7 bra 	BB0_7;

	rem.s32 	%r58, %r12, %r32;
	add.s32 	%r59, %r58, %r32;
	rem.s32 	%r123, %r59, %r32;
	bra.uni 	BB0_8;

BB0_7:
	add.s32 	%r60, %r32, -1;
	min.s32 	%r123, %r12, %r60;

BB0_8:
	add.s32 	%r61, %r123, %r5;
	cvt.s64.s32	%rd29, %r61;
	mul.wide.s32 	%rd30, %r61, 4;
	add.s64 	%rd31, %rd5, %rd30;
	add.s64 	%rd32, %rd4, %rd30;
	add.s64 	%rd33, %rd3, %rd30;
	ld.global.nc.f32 	%f50, [%rd31];
	ld.global.nc.f32 	%f51, [%rd32];
	mul.f32 	%f52, %f51, %f51;
	fma.rn.f32 	%f53, %f50, %f50, %f52;
	ld.global.nc.f32 	%f54, [%rd33];
	fma.rn.f32 	%f55, %f54, %f54, %f53;
	setp.eq.f32	%p11, %f55, 0f00000000;
	selp.f32	%f56, %f1, %f50, %p11;
	selp.f32	%f57, %f2, %f51, %p11;
	selp.f32	%f58, %f3, %f54, %p11;
	add.s64 	%rd34, %rd2, %rd29;
	ld.global.nc.u8 	%rs9, [%rd34];
	setp.gt.u16	%p12, %rs9, %rs1;
	cvt.u32.u16	%r62, %rs9;
	and.b32  	%r63, %r62, 255;
	selp.b32	%r64, %r7, %r63, %p12;
	selp.b32	%r65, %r63, %r7, %p12;
	add.s32 	%r66, %r65, 1;
	mul.lo.s32 	%r67, %r66, %r65;
	shr.u32 	%r68, %r67, 1;
	add.s32 	%r69, %r68, %r64;
	mul.wide.s32 	%rd35, %r69, 4;
	add.s64 	%rd36, %rd1, %rd35;
	ld.global.nc.f32 	%f59, [%rd36];
	mul.f32 	%f60, %f59, %f30;
	sub.f32 	%f61, %f56, %f1;
	sub.f32 	%f62, %f57, %f2;
	sub.f32 	%f63, %f58, %f3;
	fma.rn.f32 	%f7, %f60, %f61, %f4;
	fma.rn.f32 	%f8, %f60, %f62, %f5;
	fma.rn.f32 	%f9, %f60, %f63, %f6;
	and.b16  	%rs3, %rs5, 2;
	setp.eq.s16	%p13, %rs3, 0;
	add.s32 	%r16, %r2, -1;
	@%p13 bra 	BB0_10;

	rem.s32 	%r70, %r16, %r33;
	add.s32 	%r71, %r70, %r33;
	rem.s32 	%r124, %r71, %r33;
	bra.uni 	BB0_11;

BB0_10:
	mov.u32 	%r72, 0;
	max.s32 	%r124, %r16, %r72;

BB0_11:
	add.s32 	%r73, %r124, %r4;
	mad.lo.s32 	%r74, %r73, %r32, %r1;
	cvt.s64.s32	%rd37, %r74;
	mul.wide.s32 	%rd38, %r74, 4;
	add.s64 	%rd39, %rd5, %rd38;
	add.s64 	%rd40, %rd4, %rd38;
	add.s64 	%rd41, %rd3, %rd38;
	ld.global.nc.f32 	%f64, [%rd39];
	ld.global.nc.f32 	%f65, [%rd40];
	mul.f32 	%f66, %f65, %f65;
	fma.rn.f32 	%f67, %f64, %f64, %f66;
	ld.global.nc.f32 	%f68, [%rd41];
	fma.rn.f32 	%f69, %f68, %f68, %f67;
	setp.eq.f32	%p14, %f69, 0f00000000;
	selp.f32	%f70, %f1, %f64, %p14;
	selp.f32	%f71, %f2, %f65, %p14;
	selp.f32	%f72, %f3, %f68, %p14;
	add.s64 	%rd42, %rd2, %rd37;
	ld.global.nc.u8 	%rs12, [%rd42];
	setp.gt.u16	%p15, %rs12, %rs1;
	cvt.u32.u16	%r75, %rs12;
	and.b32  	%r76, %r75, 255;
	selp.b32	%r77, %r7, %r76, %p15;
	selp.b32	%r78, %r76, %r7, %p15;
	add.s32 	%r79, %r78, 1;
	mul.lo.s32 	%r80, %r79, %r78;
	shr.u32 	%r81, %r80, 1;
	add.s32 	%r82, %r81, %r77;
	mul.wide.s32 	%rd43, %r82, 4;
	add.s64 	%rd44, %rd1, %rd43;
	ld.global.nc.f32 	%f73, [%rd44];
	mul.f32 	%f74, %f73, %f31;
	sub.f32 	%f75, %f70, %f1;
	sub.f32 	%f76, %f71, %f2;
	sub.f32 	%f77, %f72, %f3;
	fma.rn.f32 	%f10, %f74, %f75, %f7;
	fma.rn.f32 	%f11, %f74, %f76, %f8;
	fma.rn.f32 	%f12, %f74, %f77, %f9;
	add.s32 	%r20, %r2, 1;
	@%p13 bra 	BB0_13;

	rem.s32 	%r83, %r20, %r33;
	add.s32 	%r84, %r83, %r33;
	rem.s32 	%r125, %r84, %r33;
	bra.uni 	BB0_14;

BB0_13:
	add.s32 	%r85, %r33, -1;
	min.s32 	%r125, %r20, %r85;

BB0_14:
	add.s32 	%r86, %r125, %r4;
	mad.lo.s32 	%r87, %r86, %r32, %r1;
	cvt.s64.s32	%rd45, %r87;
	mul.wide.s32 	%rd46, %r87, 4;
	add.s64 	%rd47, %rd5, %rd46;
	add.s64 	%rd48, %rd4, %rd46;
	add.s64 	%rd49, %rd3, %rd46;
	ld.global.nc.f32 	%f78, [%rd47];
	ld.global.nc.f32 	%f79, [%rd48];
	mul.f32 	%f80, %f79, %f79;
	fma.rn.f32 	%f81, %f78, %f78, %f80;
	ld.global.nc.f32 	%f82, [%rd49];
	fma.rn.f32 	%f83, %f82, %f82, %f81;
	setp.eq.f32	%p17, %f83, 0f00000000;
	selp.f32	%f84, %f1, %f78, %p17;
	selp.f32	%f85, %f2, %f79, %p17;
	selp.f32	%f86, %f3, %f82, %p17;
	add.s64 	%rd50, %rd2, %rd45;
	ld.global.nc.u8 	%rs16, [%rd50];
	setp.gt.u16	%p18, %rs16, %rs1;
	cvt.u32.u16	%r88, %rs16;
	and.b32  	%r89, %r88, 255;
	selp.b32	%r90, %r7, %r89, %p18;
	selp.b32	%r91, %r89, %r7, %p18;
	add.s32 	%r92, %r91, 1;
	mul.lo.s32 	%r93, %r92, %r91;
	shr.u32 	%r94, %r93, 1;
	add.s32 	%r95, %r94, %r90;
	mul.wide.s32 	%rd51, %r95, 4;
	add.s64 	%rd52, %rd1, %rd51;
	ld.global.nc.f32 	%f87, [%rd52];
	mul.f32 	%f88, %f87, %f31;
	sub.f32 	%f89, %f84, %f1;
	sub.f32 	%f90, %f85, %f2;
	sub.f32 	%f91, %f86, %f3;
	fma.rn.f32 	%f128, %f88, %f89, %f10;
	fma.rn.f32 	%f129, %f88, %f90, %f11;
	fma.rn.f32 	%f130, %f88, %f91, %f12;
	setp.eq.s32	%p19, %r34, 1;
	@%p19 bra 	BB0_22;

	and.b16  	%rs4, %rs5, 4;
	setp.eq.s16	%p20, %rs4, 0;
	add.s32 	%r24, %r3, -1;
	@%p20 bra 	BB0_17;

	rem.s32 	%r96, %r24, %r34;
	add.s32 	%r97, %r96, %r34;
	rem.s32 	%r126, %r97, %r34;
	bra.uni 	BB0_18;

BB0_17:
	mov.u32 	%r98, 0;
	max.s32 	%r126, %r24, %r98;

BB0_18:
	mad.lo.s32 	%r99, %r126, %r33, %r2;
	mad.lo.s32 	%r100, %r99, %r32, %r1;
	cvt.s64.s32	%rd53, %r100;
	mul.wide.s32 	%rd54, %r100, 4;
	add.s64 	%rd55, %rd5, %rd54;
	add.s64 	%rd56, %rd4, %rd54;
	add.s64 	%rd57, %rd3, %rd54;
	ld.global.nc.f32 	%f92, [%rd55];
	ld.global.nc.f32 	%f93, [%rd56];
	mul.f32 	%f94, %f93, %f93;
	fma.rn.f32 	%f95, %f92, %f92, %f94;
	ld.global.nc.f32 	%f96, [%rd57];
	fma.rn.f32 	%f97, %f96, %f96, %f95;
	setp.eq.f32	%p21, %f97, 0f00000000;
	selp.f32	%f98, %f1, %f92, %p21;
	selp.f32	%f99, %f2, %f93, %p21;
	selp.f32	%f100, %f3, %f96, %p21;
	add.s64 	%rd58, %rd2, %rd53;
	ld.global.nc.u8 	%rs19, [%rd58];
	setp.gt.u16	%p22, %rs19, %rs1;
	cvt.u32.u16	%r101, %rs19;
	and.b32  	%r102, %r101, 255;
	selp.b32	%r103, %r7, %r102, %p22;
	selp.b32	%r104, %r102, %r7, %p22;
	add.s32 	%r105, %r104, 1;
	mul.lo.s32 	%r106, %r105, %r104;
	shr.u32 	%r107, %r106, 1;
	add.s32 	%r108, %r107, %r103;
	mul.wide.s32 	%rd59, %r108, 4;
	add.s64 	%rd60, %rd1, %rd59;
	ld.global.nc.f32 	%f101, [%rd60];
	mul.f32 	%f102, %f101, %f32;
	sub.f32 	%f103, %f98, %f1;
	sub.f32 	%f104, %f99, %f2;
	sub.f32 	%f105, %f100, %f3;
	fma.rn.f32 	%f16, %f102, %f103, %f128;
	fma.rn.f32 	%f17, %f102, %f104, %f129;
	fma.rn.f32 	%f18, %f102, %f105, %f130;
	add.s32 	%r28, %r3, 1;
	@%p20 bra 	BB0_20;

	rem.s32 	%r109, %r28, %r34;
	add.s32 	%r110, %r109, %r34;
	rem.s32 	%r127, %r110, %r34;
	bra.uni 	BB0_21;

BB0_20:
	add.s32 	%r111, %r34, -1;
	min.s32 	%r127, %r28, %r111;

BB0_21:
	mad.lo.s32 	%r112, %r127, %r33, %r2;
	mad.lo.s32 	%r113, %r112, %r32, %r1;
	cvt.s64.s32	%rd61, %r113;
	mul.wide.s32 	%rd62, %r113, 4;
	add.s64 	%rd63, %rd5, %rd62;
	add.s64 	%rd64, %rd4, %rd62;
	add.s64 	%rd65, %rd3, %rd62;
	ld.global.nc.f32 	%f106, [%rd63];
	ld.global.nc.f32 	%f107, [%rd64];
	mul.f32 	%f108, %f107, %f107;
	fma.rn.f32 	%f109, %f106, %f106, %f108;
	ld.global.nc.f32 	%f110, [%rd65];
	fma.rn.f32 	%f111, %f110, %f110, %f109;
	setp.eq.f32	%p24, %f111, 0f00000000;
	selp.f32	%f112, %f1, %f106, %p24;
	selp.f32	%f113, %f2, %f107, %p24;
	selp.f32	%f114, %f3, %f110, %p24;
	add.s64 	%rd66, %rd2, %rd61;
	ld.global.nc.u8 	%rs23, [%rd66];
	setp.gt.u16	%p25, %rs23, %rs1;
	cvt.u32.u16	%r114, %rs23;
	and.b32  	%r115, %r114, 255;
	selp.b32	%r116, %r7, %r115, %p25;
	selp.b32	%r117, %r115, %r7, %p25;
	add.s32 	%r118, %r117, 1;
	mul.lo.s32 	%r119, %r118, %r117;
	shr.u32 	%r120, %r119, 1;
	add.s32 	%r121, %r120, %r116;
	mul.wide.s32 	%rd67, %r121, 4;
	add.s64 	%rd68, %rd1, %rd67;
	ld.global.nc.f32 	%f115, [%rd68];
	mul.f32 	%f116, %f115, %f32;
	sub.f32 	%f117, %f112, %f1;
	sub.f32 	%f118, %f113, %f2;
	sub.f32 	%f119, %f114, %f3;
	fma.rn.f32 	%f128, %f116, %f117, %f16;
	fma.rn.f32 	%f129, %f116, %f118, %f17;
	fma.rn.f32 	%f130, %f116, %f119, %f18;

BB0_22:
	setp.eq.s64	%p26, %rd9, 0;
	@%p26 bra 	BB0_24;

	cvta.to.global.u64 	%rd69, %rd9;
	add.s64 	%rd71, %rd69, %rd15;
	ld.global.nc.f32 	%f120, [%rd71];
	mul.f32 	%f131, %f120, %f131;

BB0_24:
	setp.eq.f32	%p27, %f131, 0f00000000;
	mov.f32 	%f132, 0f00000000;
	@%p27 bra 	BB0_26;

	rcp.rn.f32 	%f132, %f131;

BB0_26:
	cvta.to.global.u64 	%rd72, %rd6;
	add.s64 	%rd74, %rd72, %rd15;
	ld.global.f32 	%f122, [%rd74];
	fma.rn.f32 	%f123, %f128, %f132, %f122;
	st.global.f32 	[%rd74], %f123;
	cvta.to.global.u64 	%rd75, %rd7;
	add.s64 	%rd76, %rd75, %rd15;
	ld.global.f32 	%f124, [%rd76];
	fma.rn.f32 	%f125, %f129, %f132, %f124;
	st.global.f32 	[%rd76], %f125;
	cvta.to.global.u64 	%rd77, %rd8;
	add.s64 	%rd78, %rd77, %rd15;
	ld.global.f32 	%f126, [%rd78];
	fma.rn.f32 	%f127, %f130, %f132, %f126;
	st.global.f32 	[%rd78], %f127;

BB0_27:
	ret;
}


`
)
