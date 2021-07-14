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
.version 7.1
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
	.reg .pred 	%p<19>;
	.reg .b16 	%rs<26>;
	.reg .f32 	%f<15>;
	.reg .b32 	%r<215>;
	.reg .b64 	%rd<46>;


	ld.param.u64 	%rd1, [exchangedecode_param_0];
	ld.param.u64 	%rd2, [exchangedecode_param_1];
	ld.param.u64 	%rd3, [exchangedecode_param_2];
	ld.param.u32 	%r26, [exchangedecode_param_6];
	ld.param.u32 	%r27, [exchangedecode_param_7];
	ld.param.u32 	%r28, [exchangedecode_param_8];
	ld.param.u8 	%rs5, [exchangedecode_param_9];
	mov.u32 	%r29, %ctaid.x;
	mov.u32 	%r30, %ntid.x;
	mov.u32 	%r31, %tid.x;
	mad.lo.s32 	%r32, %r30, %r29, %r31;
	mov.u32 	%r33, %ntid.y;
	mov.u32 	%r34, %ctaid.y;
	mov.u32 	%r35, %tid.y;
	mad.lo.s32 	%r36, %r33, %r34, %r35;
	mov.u32 	%r37, %ntid.z;
	mov.u32 	%r38, %ctaid.z;
	mov.u32 	%r39, %tid.z;
	mad.lo.s32 	%r40, %r37, %r38, %r39;
	setp.ge.s32	%p1, %r36, %r27;
	setp.ge.s32	%p2, %r32, %r26;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32	%p4, %r40, %r28;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	BB0_22;

	mad.lo.s32 	%r49, %r40, %r27, %r36;
	mul.lo.s32 	%r1, %r49, %r26;
	add.s32 	%r54, %r1, %r32;
	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32	%rd5, %r54;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.nc.u8 	%rs1, [%rd6];
	and.b16  	%rs2, %rs5, 1;
	setp.eq.s16	%p6, %rs2, 0;
	add.s32 	%r2, %r32, -1;
	@%p6 bra 	BB0_3;

	rem.s32 	%r55, %r2, %r26;
	add.s32 	%r56, %r55, %r26;
	rem.s32 	%r209, %r56, %r26;
	bra.uni 	BB0_4;

BB0_3:
	mov.u32 	%r57, 0;
	max.s32 	%r209, %r2, %r57;

BB0_4:
	add.s32 	%r58, %r209, %r1;
	cvt.s64.s32	%rd8, %r58;
	add.s64 	%rd9, %rd4, %rd8;
	ld.global.nc.u8 	%rs6, [%rd9];
	setp.gt.u16	%p7, %rs6, %rs1;
	cvt.u32.u16	%r59, %rs6;
	and.b32  	%r60, %r59, 255;
	cvt.u32.u16	%r61, %rs1;
	and.b32  	%r62, %r61, 255;
	selp.b32	%r63, %r62, %r60, %p7;
	selp.b32	%r64, %r60, %r62, %p7;
	add.s32 	%r65, %r64, 1;
	mul.lo.s32 	%r66, %r65, %r64;
	shr.u32 	%r67, %r66, 1;
	add.s32 	%r68, %r67, %r63;
	cvta.to.global.u64 	%rd10, %rd2;
	mul.wide.s32 	%rd11, %r68, 4;
	add.s64 	%rd12, %rd10, %rd11;
	ld.global.nc.f32 	%f1, [%rd12];
	add.s32 	%r6, %r32, 1;
	@%p6 bra 	BB0_6;

	rem.s32 	%r73, %r6, %r26;
	add.s32 	%r74, %r73, %r26;
	rem.s32 	%r210, %r74, %r26;
	bra.uni 	BB0_7;

BB0_6:
	add.s32 	%r75, %r26, -1;
	min.s32 	%r210, %r6, %r75;

BB0_7:
	add.s32 	%r76, %r210, %r1;
	cvt.s64.s32	%rd14, %r76;
	add.s64 	%rd15, %rd4, %rd14;
	ld.global.nc.u8 	%rs9, [%rd15];
	setp.gt.u16	%p9, %rs9, %rs1;
	cvt.u32.u16	%r77, %rs9;
	and.b32  	%r78, %r77, 255;
	selp.b32	%r81, %r62, %r78, %p9;
	selp.b32	%r82, %r78, %r62, %p9;
	add.s32 	%r83, %r82, 1;
	mul.lo.s32 	%r84, %r83, %r82;
	shr.u32 	%r85, %r84, 1;
	add.s32 	%r86, %r85, %r81;
	mul.wide.s32 	%rd17, %r86, 4;
	add.s64 	%rd18, %rd10, %rd17;
	ld.global.nc.f32 	%f8, [%rd18];
	add.f32 	%f9, %f1, 0f00000000;
	add.f32 	%f2, %f9, %f8;
	and.b16  	%rs3, %rs5, 2;
	setp.eq.s16	%p10, %rs3, 0;
	add.s32 	%r10, %r36, -1;
	@%p10 bra 	BB0_9;

	rem.s32 	%r91, %r10, %r27;
	add.s32 	%r92, %r91, %r27;
	rem.s32 	%r211, %r92, %r27;
	bra.uni 	BB0_10;

BB0_9:
	mov.u32 	%r93, 0;
	max.s32 	%r211, %r10, %r93;

BB0_10:
	mad.lo.s32 	%r98, %r40, %r27, %r211;
	mad.lo.s32 	%r103, %r98, %r26, %r32;
	cvt.s64.s32	%rd20, %r103;
	add.s64 	%rd21, %rd4, %rd20;
	ld.global.nc.u8 	%rs12, [%rd21];
	setp.gt.u16	%p11, %rs12, %rs1;
	cvt.u32.u16	%r104, %rs12;
	and.b32  	%r105, %r104, 255;
	selp.b32	%r108, %r62, %r105, %p11;
	selp.b32	%r109, %r105, %r62, %p11;
	add.s32 	%r110, %r109, 1;
	mul.lo.s32 	%r111, %r110, %r109;
	shr.u32 	%r112, %r111, 1;
	add.s32 	%r113, %r112, %r108;
	mul.wide.s32 	%rd23, %r113, 4;
	add.s64 	%rd24, %rd10, %rd23;
	ld.global.nc.f32 	%f10, [%rd24];
	add.f32 	%f3, %f2, %f10;
	add.s32 	%r14, %r36, 1;
	@%p10 bra 	BB0_12;

	rem.s32 	%r118, %r14, %r27;
	add.s32 	%r119, %r118, %r27;
	rem.s32 	%r212, %r119, %r27;
	bra.uni 	BB0_13;

BB0_12:
	add.s32 	%r120, %r27, -1;
	min.s32 	%r212, %r14, %r120;

BB0_13:
	mad.lo.s32 	%r125, %r40, %r27, %r212;
	mad.lo.s32 	%r130, %r125, %r26, %r32;
	cvt.s64.s32	%rd26, %r130;
	add.s64 	%rd27, %rd4, %rd26;
	ld.global.nc.u8 	%rs16, [%rd27];
	setp.gt.u16	%p13, %rs16, %rs1;
	cvt.u32.u16	%r131, %rs16;
	and.b32  	%r132, %r131, 255;
	selp.b32	%r135, %r62, %r132, %p13;
	selp.b32	%r136, %r132, %r62, %p13;
	add.s32 	%r137, %r136, 1;
	mul.lo.s32 	%r138, %r137, %r136;
	shr.u32 	%r139, %r138, 1;
	add.s32 	%r140, %r139, %r135;
	mul.wide.s32 	%rd29, %r140, 4;
	add.s64 	%rd30, %rd10, %rd29;
	ld.global.nc.f32 	%f11, [%rd30];
	add.f32 	%f14, %f3, %f11;
	setp.eq.s32	%p14, %r28, 1;
	@%p14 bra 	BB0_21;

	and.b16  	%rs4, %rs5, 4;
	setp.eq.s16	%p15, %rs4, 0;
	add.s32 	%r18, %r40, -1;
	@%p15 bra 	BB0_16;

	rem.s32 	%r145, %r18, %r28;
	add.s32 	%r146, %r145, %r28;
	rem.s32 	%r213, %r146, %r28;
	bra.uni 	BB0_17;

BB0_16:
	mov.u32 	%r147, 0;
	max.s32 	%r213, %r18, %r147;

BB0_17:
	mad.lo.s32 	%r152, %r213, %r27, %r36;
	mad.lo.s32 	%r157, %r152, %r26, %r32;
	cvt.s64.s32	%rd32, %r157;
	add.s64 	%rd33, %rd4, %rd32;
	ld.global.nc.u8 	%rs19, [%rd33];
	setp.gt.u16	%p16, %rs19, %rs1;
	cvt.u32.u16	%r158, %rs19;
	and.b32  	%r159, %r158, 255;
	selp.b32	%r162, %r62, %r159, %p16;
	selp.b32	%r163, %r159, %r62, %p16;
	add.s32 	%r164, %r163, 1;
	mul.lo.s32 	%r165, %r164, %r163;
	shr.u32 	%r166, %r165, 1;
	add.s32 	%r167, %r166, %r162;
	mul.wide.s32 	%rd35, %r167, 4;
	add.s64 	%rd36, %rd10, %rd35;
	ld.global.nc.f32 	%f12, [%rd36];
	add.f32 	%f5, %f14, %f12;
	add.s32 	%r22, %r40, 1;
	@%p15 bra 	BB0_19;

	rem.s32 	%r172, %r22, %r28;
	add.s32 	%r173, %r172, %r28;
	rem.s32 	%r214, %r173, %r28;
	bra.uni 	BB0_20;

BB0_19:
	add.s32 	%r174, %r28, -1;
	min.s32 	%r214, %r22, %r174;

BB0_20:
	mad.lo.s32 	%r179, %r214, %r27, %r36;
	mad.lo.s32 	%r184, %r179, %r26, %r32;
	cvt.s64.s32	%rd38, %r184;
	add.s64 	%rd39, %rd4, %rd38;
	ld.global.nc.u8 	%rs23, [%rd39];
	setp.gt.u16	%p18, %rs23, %rs1;
	cvt.u32.u16	%r185, %rs23;
	and.b32  	%r186, %r185, 255;
	selp.b32	%r189, %r62, %r186, %p18;
	selp.b32	%r190, %r186, %r62, %p18;
	add.s32 	%r191, %r190, 1;
	mul.lo.s32 	%r192, %r191, %r190;
	shr.u32 	%r193, %r192, 1;
	add.s32 	%r194, %r193, %r189;
	mul.wide.s32 	%rd41, %r194, 4;
	add.s64 	%rd42, %rd10, %rd41;
	ld.global.nc.f32 	%f13, [%rd42];
	add.f32 	%f14, %f5, %f13;

BB0_21:
	mad.lo.s32 	%r208, %r49, %r26, %r32;
	cvta.to.global.u64 	%rd43, %rd1;
	mul.wide.s32 	%rd44, %r208, 4;
	add.s64 	%rd45, %rd43, %rd44;
	st.global.f32 	[%rd45], %f14;

BB0_22:
	ret;
}


`
)
