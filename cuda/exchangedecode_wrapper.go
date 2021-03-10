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

// CUDA handle for exchangedecode kernel
var exchangedecode_code cu.Function

// Stores the arguments for exchangedecode kernel invocation
type exchangedecode_args_t struct {
	arg_dst     unsafe.Pointer
	arg_aLUT2d  unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_wx      float32
	arg_wy      float32
	arg_wz      float32
	arg_Nx      int
	arg_Ny      int
	arg_Nz      int
	arg_PBC     byte
	argptr      [10]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for exchangedecode kernel invocation
var exchangedecode_args exchangedecode_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	exchangedecode_args.argptr[0] = unsafe.Pointer(&exchangedecode_args.arg_dst)
	exchangedecode_args.argptr[1] = unsafe.Pointer(&exchangedecode_args.arg_aLUT2d)
	exchangedecode_args.argptr[2] = unsafe.Pointer(&exchangedecode_args.arg_regions)
	exchangedecode_args.argptr[3] = unsafe.Pointer(&exchangedecode_args.arg_wx)
	exchangedecode_args.argptr[4] = unsafe.Pointer(&exchangedecode_args.arg_wy)
	exchangedecode_args.argptr[5] = unsafe.Pointer(&exchangedecode_args.arg_wz)
	exchangedecode_args.argptr[6] = unsafe.Pointer(&exchangedecode_args.arg_Nx)
	exchangedecode_args.argptr[7] = unsafe.Pointer(&exchangedecode_args.arg_Ny)
	exchangedecode_args.argptr[8] = unsafe.Pointer(&exchangedecode_args.arg_Nz)
	exchangedecode_args.argptr[9] = unsafe.Pointer(&exchangedecode_args.arg_PBC)
}

// Wrapper for exchangedecode CUDA kernel, asynchronous.
func k_exchangedecode_async(dst unsafe.Pointer, aLUT2d unsafe.Pointer, regions unsafe.Pointer, wx float32, wy float32, wz float32, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("exchangedecode")
	}

	exchangedecode_args.Lock()
	defer exchangedecode_args.Unlock()

	if exchangedecode_code == 0 {
		exchangedecode_code = fatbinLoad(exchangedecode_map, "exchangedecode")
	}

	exchangedecode_args.arg_dst = dst
	exchangedecode_args.arg_aLUT2d = aLUT2d
	exchangedecode_args.arg_regions = regions
	exchangedecode_args.arg_wx = wx
	exchangedecode_args.arg_wy = wy
	exchangedecode_args.arg_wz = wz
	exchangedecode_args.arg_Nx = Nx
	exchangedecode_args.arg_Ny = Ny
	exchangedecode_args.arg_Nz = Nz
	exchangedecode_args.arg_PBC = PBC

	args := exchangedecode_args.argptr[:]
	cu.LaunchKernel(exchangedecode_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("exchangedecode")
	}
}

// maps compute capability on PTX code for exchangedecode kernel.
var exchangedecode_map = map[int]string{0: "",
	70: exchangedecode_ptx_70}

// exchangedecode PTX code for various compute capabilities.
const (
	exchangedecode_ptx_70 = `
.version 7.2
.target sm_70
.address_size 64

	// .globl	exchangedecode

.visible .entry exchangedecode(
	.param .u64 exchangedecode_param_0,
	.param .u64 exchangedecode_param_1,
	.param .u64 exchangedecode_param_2,
	.param .f32 exchangedecode_param_3,
	.param .f32 exchangedecode_param_4,
	.param .f32 exchangedecode_param_5,
	.param .u32 exchangedecode_param_6,
	.param .u32 exchangedecode_param_7,
	.param .u32 exchangedecode_param_8,
	.param .u8 exchangedecode_param_9
)
{
	.reg .pred 	%p<13>;
	.reg .b16 	%rs<36>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<156>;
	.reg .b64 	%rd<46>;


	ld.param.u8 	%rs5, [exchangedecode_param_9];
	ld.param.u64 	%rd1, [exchangedecode_param_0];
	ld.param.u64 	%rd2, [exchangedecode_param_1];
	ld.param.u64 	%rd3, [exchangedecode_param_2];
	ld.param.u32 	%r29, [exchangedecode_param_6];
	ld.param.u32 	%r30, [exchangedecode_param_7];
	ld.param.u32 	%r31, [exchangedecode_param_8];
	mov.u32 	%r32, %ctaid.x;
	mov.u32 	%r33, %ntid.x;
	mov.u32 	%r34, %tid.x;
	mad.lo.s32 	%r1, %r32, %r33, %r34;
	mov.u32 	%r35, %ntid.y;
	mov.u32 	%r36, %ctaid.y;
	mov.u32 	%r37, %tid.y;
	mad.lo.s32 	%r2, %r36, %r35, %r37;
	mov.u32 	%r38, %ntid.z;
	mov.u32 	%r39, %ctaid.z;
	mov.u32 	%r40, %tid.z;
	mad.lo.s32 	%r3, %r39, %r38, %r40;
	setp.ge.s32 	%p1, %r1, %r29;
	setp.ge.s32 	%p2, %r2, %r30;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r31;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	LBB0_22;

	mad.lo.s32 	%r41, %r3, %r30, %r2;
	mul.lo.s32 	%r4, %r41, %r29;
	add.s32 	%r42, %r4, %r1;
	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32 	%rd5, %r42;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	and.b16  	%rs2, %rs5, 1;
	setp.eq.s16 	%p6, %rs2, 0;
	add.s32 	%r5, %r1, -1;
	@%p6 bra 	LBB0_3;
	bra.uni 	LBB0_2;

LBB0_3:
	max.s32 	%r150, %r5, 0;
	bra.uni 	LBB0_4;

LBB0_2:
	rem.s32 	%r43, %r5, %r29;
	add.s32 	%r44, %r43, %r29;
	rem.s32 	%r150, %r44, %r29;

LBB0_4:
	add.s32 	%r45, %r150, %r4;
	cvt.s64.s32 	%rd8, %r45;
	add.s64 	%rd9, %rd4, %rd8;
	ld.global.nc.u8 	%rs6, [%rd9];
	min.u16 	%rs9, %rs6, %rs1;
	cvt.u32.u16 	%r46, %rs9;
	max.u16 	%rs10, %rs6, %rs1;
	cvt.u32.u16 	%r47, %rs10;
	add.s32 	%r48, %r47, 1;
	mul.lo.s32 	%r49, %r48, %r47;
	shr.u32 	%r50, %r49, 1;
	add.s32 	%r51, %r50, %r46;
	cvta.to.global.u64 	%rd10, %rd2;
	mul.wide.s32 	%rd11, %r51, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f1, [%rd12];
	add.s32 	%r9, %r1, 1;
	@%p6 bra 	LBB0_6;
	bra.uni 	LBB0_5;

LBB0_6:
	add.s32 	%r54, %r29, -1;
	min.s32 	%r151, %r9, %r54;
	bra.uni 	LBB0_7;

LBB0_5:
	rem.s32 	%r52, %r9, %r29;
	add.s32 	%r53, %r52, %r29;
	rem.s32 	%r151, %r53, %r29;

LBB0_7:
	add.s32 	%r55, %r151, %r4;
	cvt.s64.s32 	%rd14, %r55;
	add.s64 	%rd15, %rd4, %rd14;
	ld.global.nc.u8 	%rs11, [%rd15];
	min.u16 	%rs14, %rs11, %rs1;
	cvt.u32.u16 	%r56, %rs14;
	max.u16 	%rs15, %rs11, %rs1;
	cvt.u32.u16 	%r57, %rs15;
	add.s32 	%r58, %r57, 1;
	mul.lo.s32 	%r59, %r58, %r57;
	shr.u32 	%r60, %r59, 1;
	add.s32 	%r61, %r60, %r56;
	mul.wide.s32 	%rd17, %r61, 4;
	add.s64 	%rd18, %rd10, %rd17;
	ld.global.nc.f32 	%f8, [%rd18];
	add.f32 	%f9, %f1, 0f00000000;
	add.f32 	%f2, %f9, %f8;
	and.b16  	%rs3, %rs5, 2;
	setp.eq.s16 	%p8, %rs3, 0;
	add.s32 	%r13, %r2, -1;
	@%p8 bra 	LBB0_9;
	bra.uni 	LBB0_8;

LBB0_9:
	max.s32 	%r152, %r13, 0;
	bra.uni 	LBB0_10;

LBB0_8:
	rem.s32 	%r66, %r13, %r30;
	add.s32 	%r67, %r66, %r30;
	rem.s32 	%r152, %r67, %r30;

LBB0_10:
	mad.lo.s32 	%r72, %r3, %r30, %r152;
	mad.lo.s32 	%r73, %r72, %r29, %r1;
	cvt.s64.s32 	%rd20, %r73;
	add.s64 	%rd21, %rd4, %rd20;
	ld.global.nc.u8 	%rs16, [%rd21];
	min.u16 	%rs19, %rs16, %rs1;
	cvt.u32.u16 	%r74, %rs19;
	max.u16 	%rs20, %rs16, %rs1;
	cvt.u32.u16 	%r75, %rs20;
	add.s32 	%r76, %r75, 1;
	mul.lo.s32 	%r77, %r76, %r75;
	shr.u32 	%r78, %r77, 1;
	add.s32 	%r79, %r78, %r74;
	mul.wide.s32 	%rd23, %r79, 4;
	add.s64 	%rd24, %rd10, %rd23;
	ld.global.nc.f32 	%f10, [%rd24];
	add.f32 	%f3, %f2, %f10;
	add.s32 	%r17, %r2, 1;
	@%p8 bra 	LBB0_12;
	bra.uni 	LBB0_11;

LBB0_12:
	add.s32 	%r86, %r30, -1;
	min.s32 	%r153, %r17, %r86;
	bra.uni 	LBB0_13;

LBB0_11:
	rem.s32 	%r84, %r17, %r30;
	add.s32 	%r85, %r84, %r30;
	rem.s32 	%r153, %r85, %r30;

LBB0_13:
	mad.lo.s32 	%r91, %r3, %r30, %r153;
	mad.lo.s32 	%r92, %r91, %r29, %r1;
	cvt.s64.s32 	%rd26, %r92;
	add.s64 	%rd27, %rd4, %rd26;
	ld.global.nc.u8 	%rs21, [%rd27];
	min.u16 	%rs24, %rs21, %rs1;
	cvt.u32.u16 	%r93, %rs24;
	max.u16 	%rs25, %rs21, %rs1;
	cvt.u32.u16 	%r94, %rs25;
	add.s32 	%r95, %r94, 1;
	mul.lo.s32 	%r96, %r95, %r94;
	shr.u32 	%r97, %r96, 1;
	add.s32 	%r98, %r97, %r93;
	mul.wide.s32 	%rd29, %r98, 4;
	add.s64 	%rd30, %rd10, %rd29;
	ld.global.nc.f32 	%f11, [%rd30];
	add.f32 	%f14, %f3, %f11;
	setp.eq.s32 	%p10, %r31, 1;
	@%p10 bra 	LBB0_21;

	and.b16  	%rs4, %rs5, 4;
	setp.eq.s16 	%p11, %rs4, 0;
	add.s32 	%r21, %r3, -1;
	@%p11 bra 	LBB0_16;
	bra.uni 	LBB0_15;

LBB0_16:
	max.s32 	%r154, %r21, 0;
	bra.uni 	LBB0_17;

LBB0_15:
	rem.s32 	%r103, %r21, %r31;
	add.s32 	%r104, %r103, %r31;
	rem.s32 	%r154, %r104, %r31;

LBB0_17:
	mad.lo.s32 	%r109, %r154, %r30, %r2;
	mad.lo.s32 	%r110, %r109, %r29, %r1;
	cvt.s64.s32 	%rd32, %r110;
	add.s64 	%rd33, %rd4, %rd32;
	ld.global.nc.u8 	%rs26, [%rd33];
	min.u16 	%rs29, %rs26, %rs1;
	cvt.u32.u16 	%r111, %rs29;
	max.u16 	%rs30, %rs26, %rs1;
	cvt.u32.u16 	%r112, %rs30;
	add.s32 	%r113, %r112, 1;
	mul.lo.s32 	%r114, %r113, %r112;
	shr.u32 	%r115, %r114, 1;
	add.s32 	%r116, %r115, %r111;
	mul.wide.s32 	%rd35, %r116, 4;
	add.s64 	%rd36, %rd10, %rd35;
	ld.global.nc.f32 	%f12, [%rd36];
	add.f32 	%f5, %f14, %f12;
	add.s32 	%r25, %r3, 1;
	@%p11 bra 	LBB0_19;
	bra.uni 	LBB0_18;

LBB0_19:
	add.s32 	%r123, %r31, -1;
	min.s32 	%r155, %r25, %r123;
	bra.uni 	LBB0_20;

LBB0_18:
	rem.s32 	%r121, %r25, %r31;
	add.s32 	%r122, %r121, %r31;
	rem.s32 	%r155, %r122, %r31;

LBB0_20:
	mad.lo.s32 	%r128, %r155, %r30, %r2;
	mad.lo.s32 	%r129, %r128, %r29, %r1;
	cvt.s64.s32 	%rd38, %r129;
	add.s64 	%rd39, %rd4, %rd38;
	ld.global.nc.u8 	%rs31, [%rd39];
	min.u16 	%rs34, %rs31, %rs1;
	cvt.u32.u16 	%r130, %rs34;
	max.u16 	%rs35, %rs31, %rs1;
	cvt.u32.u16 	%r131, %rs35;
	add.s32 	%r132, %r131, 1;
	mul.lo.s32 	%r133, %r132, %r131;
	shr.u32 	%r134, %r133, 1;
	add.s32 	%r135, %r134, %r130;
	mul.wide.s32 	%rd41, %r135, 4;
	add.s64 	%rd42, %rd10, %rd41;
	ld.global.nc.f32 	%f13, [%rd42];
	add.f32 	%f14, %f5, %f13;

LBB0_21:
	mad.lo.s32 	%r149, %r41, %r29, %r1;
	cvta.to.global.u64 	%rd43, %rd1;
	mul.wide.s32 	%rd44, %r149, 4;
	add.s64 	%rd45, %rd43, %rd44;
	st.global.f32 	[%rd45], %f14;

LBB0_22:
	ret;

}

`
)
