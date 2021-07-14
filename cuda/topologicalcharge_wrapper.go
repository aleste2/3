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

// CUDA handle for settopologicalcharge kernel
var settopologicalcharge_code cu.Function

// Stores the arguments for settopologicalcharge kernel invocation
type settopologicalcharge_args_t struct {
	arg_s     unsafe.Pointer
	arg_mx    unsafe.Pointer
	arg_my    unsafe.Pointer
	arg_mz    unsafe.Pointer
	arg_icxcy float32
	arg_Nx    int
	arg_Ny    int
	arg_Nz    int
	arg_PBC   byte
	argptr    [9]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for settopologicalcharge kernel invocation
var settopologicalcharge_args settopologicalcharge_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	settopologicalcharge_args.argptr[0] = unsafe.Pointer(&settopologicalcharge_args.arg_s)
	settopologicalcharge_args.argptr[1] = unsafe.Pointer(&settopologicalcharge_args.arg_mx)
	settopologicalcharge_args.argptr[2] = unsafe.Pointer(&settopologicalcharge_args.arg_my)
	settopologicalcharge_args.argptr[3] = unsafe.Pointer(&settopologicalcharge_args.arg_mz)
	settopologicalcharge_args.argptr[4] = unsafe.Pointer(&settopologicalcharge_args.arg_icxcy)
	settopologicalcharge_args.argptr[5] = unsafe.Pointer(&settopologicalcharge_args.arg_Nx)
	settopologicalcharge_args.argptr[6] = unsafe.Pointer(&settopologicalcharge_args.arg_Ny)
	settopologicalcharge_args.argptr[7] = unsafe.Pointer(&settopologicalcharge_args.arg_Nz)
	settopologicalcharge_args.argptr[8] = unsafe.Pointer(&settopologicalcharge_args.arg_PBC)
}

// Wrapper for settopologicalcharge CUDA kernel, asynchronous.
func k_settopologicalcharge_async(s unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, icxcy float32, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("settopologicalcharge")
	}

	settopologicalcharge_args.Lock()
	defer settopologicalcharge_args.Unlock()

	if settopologicalcharge_code == 0 {
		settopologicalcharge_code = fatbinLoad(settopologicalcharge_map, "settopologicalcharge")
	}

	settopologicalcharge_args.arg_s = s
	settopologicalcharge_args.arg_mx = mx
	settopologicalcharge_args.arg_my = my
	settopologicalcharge_args.arg_mz = mz
	settopologicalcharge_args.arg_icxcy = icxcy
	settopologicalcharge_args.arg_Nx = Nx
	settopologicalcharge_args.arg_Ny = Ny
	settopologicalcharge_args.arg_Nz = Nz
	settopologicalcharge_args.arg_PBC = PBC

	args := settopologicalcharge_args.argptr[:]
	cu.LaunchKernel(settopologicalcharge_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("settopologicalcharge")
	}
}

// maps compute capability on PTX code for settopologicalcharge kernel.
var settopologicalcharge_map = map[int]string{0: "",
	70: settopologicalcharge_ptx_70}

// settopologicalcharge PTX code for various compute capabilities.
const (
	settopologicalcharge_ptx_70 = `
.version 7.1
.target sm_70
.address_size 64

	// .globl	settopologicalcharge

.visible .entry settopologicalcharge(
	.param .u64 settopologicalcharge_param_0,
	.param .u64 settopologicalcharge_param_1,
	.param .u64 settopologicalcharge_param_2,
	.param .u64 settopologicalcharge_param_3,
	.param .f32 settopologicalcharge_param_4,
	.param .u32 settopologicalcharge_param_5,
	.param .u32 settopologicalcharge_param_6,
	.param .u32 settopologicalcharge_param_7,
	.param .u8 settopologicalcharge_param_8
)
{
	.reg .pred 	%p<77>;
	.reg .b16 	%rs<11>;
	.reg .f32 	%f<265>;
	.reg .b32 	%r<97>;
	.reg .b64 	%rd<46>;


	ld.param.u64 	%rd5, [settopologicalcharge_param_0];
	ld.param.u64 	%rd6, [settopologicalcharge_param_1];
	ld.param.u64 	%rd7, [settopologicalcharge_param_2];
	ld.param.u64 	%rd8, [settopologicalcharge_param_3];
	ld.param.f32 	%f102, [settopologicalcharge_param_4];
	ld.param.u32 	%r40, [settopologicalcharge_param_5];
	ld.param.u32 	%r41, [settopologicalcharge_param_6];
	ld.param.u32 	%r42, [settopologicalcharge_param_7];
	ld.param.u8 	%rs3, [settopologicalcharge_param_8];
	cvta.to.global.u64 	%rd1, %rd8;
	cvta.to.global.u64 	%rd2, %rd7;
	cvta.to.global.u64 	%rd3, %rd6;
	mov.u32 	%r43, %ntid.x;
	mov.u32 	%r44, %ctaid.x;
	mov.u32 	%r45, %tid.x;
	mad.lo.s32 	%r1, %r43, %r44, %r45;
	mov.u32 	%r46, %ntid.y;
	mov.u32 	%r47, %ctaid.y;
	mov.u32 	%r48, %tid.y;
	mad.lo.s32 	%r2, %r46, %r47, %r48;
	mov.u32 	%r49, %ntid.z;
	mov.u32 	%r50, %ctaid.z;
	mov.u32 	%r51, %tid.z;
	mad.lo.s32 	%r3, %r49, %r50, %r51;
	setp.ge.s32	%p3, %r2, %r41;
	setp.ge.s32	%p4, %r1, %r40;
	or.pred  	%p5, %p3, %p4;
	setp.ge.s32	%p6, %r3, %r42;
	or.pred  	%p7, %p5, %p6;
	@%p7 bra 	BB0_72;

	mul.lo.s32 	%r4, %r3, %r41;
	add.s32 	%r52, %r4, %r2;
	mul.lo.s32 	%r5, %r52, %r40;
	add.s32 	%r53, %r5, %r1;
	mul.wide.s32 	%rd9, %r53, 4;
	add.s64 	%rd10, %rd3, %rd9;
	add.s64 	%rd11, %rd2, %rd9;
	add.s64 	%rd12, %rd1, %rd9;
	ld.global.nc.f32 	%f1, [%rd10];
	ld.global.nc.f32 	%f2, [%rd11];
	mul.f32 	%f103, %f2, %f2;
	fma.rn.f32 	%f104, %f1, %f1, %f103;
	ld.global.nc.f32 	%f3, [%rd12];
	fma.rn.f32 	%f105, %f3, %f3, %f104;
	setp.eq.f32	%p8, %f105, 0f00000000;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd4, %rd13, %rd9;
	@%p8 bra 	BB0_71;
	bra.uni 	BB0_2;

BB0_71:
	mov.u32 	%r88, 0;
	st.global.u32 	[%rd4], %r88;
	bra.uni 	BB0_72;

BB0_2:
	and.b16  	%rs1, %rs3, 1;
	setp.eq.s16	%p9, %rs1, 0;
	add.s32 	%r6, %r1, -2;
	@%p9 bra 	BB0_4;

	rem.s32 	%r54, %r6, %r40;
	add.s32 	%r55, %r54, %r40;
	rem.s32 	%r89, %r55, %r40;
	bra.uni 	BB0_5;

BB0_4:
	mov.u32 	%r56, 0;
	max.s32 	%r89, %r6, %r56;

BB0_5:
	setp.lt.s32	%p11, %r6, 0;
	mov.f32 	%f7, 0f00000000;
	and.pred  	%p12, %p11, %p9;
	mov.f32 	%f8, %f7;
	mov.f32 	%f9, %f7;
	@%p12 bra 	BB0_7;

	add.s32 	%r57, %r89, %r5;
	mul.wide.s32 	%rd14, %r57, 4;
	add.s64 	%rd15, %rd3, %rd14;
	ld.global.nc.f32 	%f7, [%rd15];
	add.s64 	%rd16, %rd2, %rd14;
	ld.global.nc.f32 	%f8, [%rd16];
	add.s64 	%rd17, %rd1, %rd14;
	ld.global.nc.f32 	%f9, [%rd17];

BB0_7:
	add.s32 	%r10, %r1, -1;
	@%p9 bra 	BB0_9;

	rem.s32 	%r58, %r10, %r40;
	add.s32 	%r59, %r58, %r40;
	rem.s32 	%r90, %r59, %r40;
	bra.uni 	BB0_10;

BB0_9:
	mov.u32 	%r60, 0;
	max.s32 	%r90, %r10, %r60;

BB0_10:
	setp.lt.s32	%p14, %r10, 0;
	mov.f32 	%f13, 0f00000000;
	and.pred  	%p16, %p14, %p9;
	mov.f32 	%f14, %f13;
	mov.f32 	%f15, %f13;
	@%p16 bra 	BB0_12;

	add.s32 	%r61, %r90, %r5;
	mul.wide.s32 	%rd18, %r61, 4;
	add.s64 	%rd19, %rd3, %rd18;
	ld.global.nc.f32 	%f13, [%rd19];
	add.s64 	%rd20, %rd2, %rd18;
	ld.global.nc.f32 	%f14, [%rd20];
	add.s64 	%rd21, %rd1, %rd18;
	ld.global.nc.f32 	%f15, [%rd21];

BB0_12:
	add.s32 	%r14, %r1, 1;
	@%p9 bra 	BB0_14;

	rem.s32 	%r62, %r14, %r40;
	add.s32 	%r63, %r62, %r40;
	rem.s32 	%r91, %r63, %r40;
	bra.uni 	BB0_15;

BB0_14:
	add.s32 	%r64, %r40, -1;
	min.s32 	%r91, %r14, %r64;

BB0_15:
	setp.ge.s32	%p18, %r14, %r40;
	mov.f32 	%f19, 0f00000000;
	and.pred  	%p20, %p18, %p9;
	mov.f32 	%f20, %f19;
	mov.f32 	%f21, %f19;
	@%p20 bra 	BB0_17;

	add.s32 	%r65, %r91, %r5;
	mul.wide.s32 	%rd22, %r65, 4;
	add.s64 	%rd23, %rd3, %rd22;
	ld.global.nc.f32 	%f19, [%rd23];
	add.s64 	%rd24, %rd2, %rd22;
	ld.global.nc.f32 	%f20, [%rd24];
	add.s64 	%rd25, %rd1, %rd22;
	ld.global.nc.f32 	%f21, [%rd25];

BB0_17:
	add.s32 	%r18, %r1, 2;
	@%p9 bra 	BB0_19;

	rem.s32 	%r66, %r18, %r40;
	add.s32 	%r67, %r66, %r40;
	rem.s32 	%r92, %r67, %r40;
	bra.uni 	BB0_20;

BB0_19:
	add.s32 	%r68, %r40, -1;
	min.s32 	%r92, %r18, %r68;

BB0_20:
	add.s32 	%r22, %r92, %r5;
	setp.ge.s32	%p22, %r18, %r40;
	mov.f32 	%f25, 0f00000000;
	and.pred  	%p24, %p22, %p9;
	mov.f32 	%f26, %f25;
	mov.f32 	%f27, %f25;
	@%p24 bra 	BB0_22;

	mul.wide.s32 	%rd26, %r22, 4;
	add.s64 	%rd27, %rd3, %rd26;
	ld.global.nc.f32 	%f25, [%rd27];
	add.s64 	%rd28, %rd2, %rd26;
	ld.global.nc.f32 	%f26, [%rd28];
	add.s64 	%rd29, %rd1, %rd26;
	ld.global.nc.f32 	%f27, [%rd29];

BB0_22:
	mul.f32 	%f118, %f20, %f20;
	fma.rn.f32 	%f119, %f19, %f19, %f118;
	fma.rn.f32 	%f28, %f21, %f21, %f119;
	setp.neu.f32	%p25, %f28, 0f00000000;
	@%p25 bra 	BB0_24;

	mul.f32 	%f123, %f14, %f14;
	fma.rn.f32 	%f124, %f13, %f13, %f123;
	fma.rn.f32 	%f125, %f15, %f15, %f124;
	setp.eq.f32	%p26, %f125, 0f00000000;
	mov.f32 	%f247, 0f00000000;
	mov.f32 	%f248, %f247;
	mov.f32 	%f249, %f247;
	@%p26 bra 	BB0_36;

BB0_24:
	mul.f32 	%f126, %f8, %f8;
	fma.rn.f32 	%f127, %f7, %f7, %f126;
	fma.rn.f32 	%f29, %f9, %f9, %f127;
	setp.eq.f32	%p27, %f29, 0f00000000;
	mul.f32 	%f128, %f26, %f26;
	fma.rn.f32 	%f129, %f25, %f25, %f128;
	fma.rn.f32 	%f30, %f27, %f27, %f129;
	setp.eq.f32	%p28, %f30, 0f00000000;
	or.pred  	%p29, %p27, %p28;
	and.pred  	%p31, %p29, %p25;
	@!%p31 bra 	BB0_26;
	bra.uni 	BB0_25;

BB0_25:
	mul.f32 	%f130, %f14, %f14;
	fma.rn.f32 	%f131, %f13, %f13, %f130;
	fma.rn.f32 	%f132, %f15, %f15, %f131;
	setp.neu.f32	%p32, %f132, 0f00000000;
	@%p32 bra 	BB0_35;
	bra.uni 	BB0_26;

BB0_35:
	sub.f32 	%f161, %f19, %f13;
	mul.f32 	%f247, %f161, 0f3F000000;
	sub.f32 	%f162, %f20, %f14;
	mul.f32 	%f248, %f162, 0f3F000000;
	sub.f32 	%f163, %f21, %f15;
	mul.f32 	%f249, %f163, 0f3F000000;
	bra.uni 	BB0_36;

BB0_26:
	setp.neu.f32	%p33, %f29, 0f00000000;
	or.pred  	%p34, %p25, %p33;
	@%p34 bra 	BB0_28;
	bra.uni 	BB0_27;

BB0_28:
	mul.f32 	%f133, %f14, %f14;
	fma.rn.f32 	%f134, %f13, %f13, %f133;
	fma.rn.f32 	%f34, %f15, %f15, %f134;
	setp.neu.f32	%p35, %f34, 0f00000000;
	setp.neu.f32	%p36, %f30, 0f00000000;
	or.pred  	%p37, %p35, %p36;
	@%p37 bra 	BB0_30;
	bra.uni 	BB0_29;

BB0_30:
	or.pred  	%p39, %p27, %p25;
	@%p39 bra 	BB0_32;
	bra.uni 	BB0_31;

BB0_32:
	or.pred  	%p42, %p28, %p35;
	@%p42 bra 	BB0_34;
	bra.uni 	BB0_33;

BB0_34:
	sub.f32 	%f152, %f19, %f13;
	sub.f32 	%f153, %f20, %f14;
	sub.f32 	%f154, %f21, %f15;
	sub.f32 	%f155, %f7, %f25;
	mul.f32 	%f156, %f155, 0f3DAAAAAB;
	sub.f32 	%f157, %f8, %f26;
	mul.f32 	%f158, %f157, 0f3DAAAAAB;
	sub.f32 	%f159, %f9, %f27;
	mul.f32 	%f160, %f159, 0f3DAAAAAB;
	fma.rn.f32 	%f247, %f152, 0f3F2AAAAB, %f156;
	fma.rn.f32 	%f248, %f153, 0f3F2AAAAB, %f158;
	fma.rn.f32 	%f249, %f154, 0f3F2AAAAB, %f160;
	bra.uni 	BB0_36;

BB0_27:
	sub.f32 	%f247, %f1, %f13;
	sub.f32 	%f248, %f2, %f14;
	sub.f32 	%f249, %f3, %f15;
	bra.uni 	BB0_36;

BB0_29:
	sub.f32 	%f247, %f19, %f1;
	sub.f32 	%f248, %f20, %f2;
	sub.f32 	%f249, %f21, %f3;
	bra.uni 	BB0_36;

BB0_31:
	mul.f32 	%f135, %f13, 0fC0000000;
	fma.rn.f32 	%f136, %f7, 0f3F000000, %f135;
	add.f32 	%f137, %f14, %f14;
	mul.f32 	%f138, %f8, 0f3F000000;
	sub.f32 	%f139, %f138, %f137;
	add.f32 	%f140, %f15, %f15;
	mul.f32 	%f141, %f9, 0f3F000000;
	sub.f32 	%f142, %f141, %f140;
	fma.rn.f32 	%f247, %f1, 0f3FC00000, %f136;
	fma.rn.f32 	%f248, %f2, 0f3FC00000, %f139;
	fma.rn.f32 	%f249, %f3, 0f3FC00000, %f142;
	bra.uni 	BB0_36;

BB0_33:
	mul.f32 	%f143, %f25, 0fBF000000;
	fma.rn.f32 	%f144, %f19, 0f40000000, %f143;
	mul.f32 	%f145, %f26, 0fBF000000;
	fma.rn.f32 	%f146, %f20, 0f40000000, %f145;
	mul.f32 	%f147, %f27, 0fBF000000;
	fma.rn.f32 	%f148, %f21, 0f40000000, %f147;
	mul.f32 	%f149, %f1, 0f3FC00000;
	sub.f32 	%f247, %f144, %f149;
	mul.f32 	%f150, %f2, 0f3FC00000;
	sub.f32 	%f248, %f146, %f150;
	mul.f32 	%f151, %f3, 0f3FC00000;
	sub.f32 	%f249, %f148, %f151;

BB0_36:
	and.b16  	%rs2, %rs3, 2;
	setp.eq.s16	%p43, %rs2, 0;
	add.s32 	%r23, %r2, -2;
	@%p43 bra 	BB0_38;

	rem.s32 	%r69, %r23, %r41;
	add.s32 	%r70, %r69, %r41;
	rem.s32 	%r93, %r70, %r41;
	bra.uni 	BB0_39;

BB0_38:
	mov.u32 	%r71, 0;
	max.s32 	%r93, %r23, %r71;

BB0_39:
	setp.lt.s32	%p45, %r23, 0;
	mov.f32 	%f56, 0f00000000;
	and.pred  	%p46, %p45, %p43;
	mov.f32 	%f57, %f56;
	mov.f32 	%f58, %f56;
	@%p46 bra 	BB0_41;

	add.s32 	%r72, %r93, %r4;
	mad.lo.s32 	%r73, %r72, %r40, %r1;
	mul.wide.s32 	%rd30, %r73, 4;
	add.s64 	%rd31, %rd3, %rd30;
	ld.global.nc.f32 	%f56, [%rd31];
	add.s64 	%rd32, %rd2, %rd30;
	ld.global.nc.f32 	%f57, [%rd32];
	add.s64 	%rd33, %rd1, %rd30;
	ld.global.nc.f32 	%f58, [%rd33];

BB0_41:
	add.s32 	%r27, %r2, -1;
	@%p43 bra 	BB0_43;

	rem.s32 	%r74, %r27, %r41;
	add.s32 	%r75, %r74, %r41;
	rem.s32 	%r94, %r75, %r41;
	bra.uni 	BB0_44;

BB0_43:
	mov.u32 	%r76, 0;
	max.s32 	%r94, %r27, %r76;

BB0_44:
	setp.lt.s32	%p48, %r27, 0;
	mov.f32 	%f62, 0f00000000;
	and.pred  	%p50, %p48, %p43;
	mov.f32 	%f63, %f62;
	mov.f32 	%f64, %f62;
	@%p50 bra 	BB0_46;

	add.s32 	%r77, %r94, %r4;
	mad.lo.s32 	%r78, %r77, %r40, %r1;
	mul.wide.s32 	%rd34, %r78, 4;
	add.s64 	%rd35, %rd3, %rd34;
	ld.global.nc.f32 	%f62, [%rd35];
	add.s64 	%rd36, %rd2, %rd34;
	ld.global.nc.f32 	%f63, [%rd36];
	add.s64 	%rd37, %rd1, %rd34;
	ld.global.nc.f32 	%f64, [%rd37];

BB0_46:
	add.s32 	%r31, %r2, 1;
	@%p43 bra 	BB0_48;

	rem.s32 	%r79, %r31, %r41;
	add.s32 	%r80, %r79, %r41;
	rem.s32 	%r95, %r80, %r41;
	bra.uni 	BB0_49;

BB0_48:
	add.s32 	%r81, %r41, -1;
	min.s32 	%r95, %r31, %r81;

BB0_49:
	setp.ge.s32	%p52, %r31, %r41;
	mov.f32 	%f68, 0f00000000;
	and.pred  	%p54, %p52, %p43;
	mov.f32 	%f69, %f68;
	mov.f32 	%f70, %f68;
	@%p54 bra 	BB0_51;

	add.s32 	%r82, %r95, %r4;
	mad.lo.s32 	%r83, %r82, %r40, %r1;
	mul.wide.s32 	%rd38, %r83, 4;
	add.s64 	%rd39, %rd3, %rd38;
	ld.global.nc.f32 	%f68, [%rd39];
	add.s64 	%rd40, %rd2, %rd38;
	ld.global.nc.f32 	%f69, [%rd40];
	add.s64 	%rd41, %rd1, %rd38;
	ld.global.nc.f32 	%f70, [%rd41];

BB0_51:
	add.s32 	%r35, %r2, 2;
	@%p43 bra 	BB0_53;

	rem.s32 	%r84, %r35, %r41;
	add.s32 	%r85, %r84, %r41;
	rem.s32 	%r96, %r85, %r41;
	bra.uni 	BB0_54;

BB0_53:
	add.s32 	%r86, %r41, -1;
	min.s32 	%r96, %r35, %r86;

BB0_54:
	add.s32 	%r87, %r96, %r4;
	mad.lo.s32 	%r39, %r87, %r40, %r1;
	setp.ge.s32	%p56, %r35, %r41;
	mov.f32 	%f74, 0f00000000;
	and.pred  	%p58, %p56, %p43;
	mov.f32 	%f75, %f74;
	mov.f32 	%f76, %f74;
	@%p58 bra 	BB0_56;

	mul.wide.s32 	%rd42, %r39, 4;
	add.s64 	%rd43, %rd3, %rd42;
	ld.global.nc.f32 	%f74, [%rd43];
	add.s64 	%rd44, %rd2, %rd42;
	ld.global.nc.f32 	%f75, [%rd44];
	add.s64 	%rd45, %rd1, %rd42;
	ld.global.nc.f32 	%f76, [%rd45];

BB0_56:
	mul.f32 	%f176, %f69, %f69;
	fma.rn.f32 	%f177, %f68, %f68, %f176;
	fma.rn.f32 	%f77, %f70, %f70, %f177;
	setp.neu.f32	%p59, %f77, 0f00000000;
	@%p59 bra 	BB0_58;

	mul.f32 	%f181, %f63, %f63;
	fma.rn.f32 	%f182, %f62, %f62, %f181;
	fma.rn.f32 	%f183, %f64, %f64, %f182;
	setp.eq.f32	%p60, %f183, 0f00000000;
	mov.f32 	%f262, 0f00000000;
	mov.f32 	%f263, %f262;
	mov.f32 	%f264, %f262;
	@%p60 bra 	BB0_70;

BB0_58:
	mul.f32 	%f184, %f57, %f57;
	fma.rn.f32 	%f185, %f56, %f56, %f184;
	fma.rn.f32 	%f78, %f58, %f58, %f185;
	setp.eq.f32	%p61, %f78, 0f00000000;
	mul.f32 	%f186, %f75, %f75;
	fma.rn.f32 	%f187, %f74, %f74, %f186;
	fma.rn.f32 	%f79, %f76, %f76, %f187;
	setp.eq.f32	%p62, %f79, 0f00000000;
	or.pred  	%p63, %p61, %p62;
	and.pred  	%p65, %p63, %p59;
	@!%p65 bra 	BB0_60;
	bra.uni 	BB0_59;

BB0_59:
	mul.f32 	%f188, %f63, %f63;
	fma.rn.f32 	%f189, %f62, %f62, %f188;
	fma.rn.f32 	%f190, %f64, %f64, %f189;
	setp.neu.f32	%p66, %f190, 0f00000000;
	@%p66 bra 	BB0_69;
	bra.uni 	BB0_60;

BB0_69:
	sub.f32 	%f219, %f68, %f62;
	mul.f32 	%f262, %f219, 0f3F000000;
	sub.f32 	%f220, %f69, %f63;
	mul.f32 	%f263, %f220, 0f3F000000;
	sub.f32 	%f221, %f70, %f64;
	mul.f32 	%f264, %f221, 0f3F000000;
	bra.uni 	BB0_70;

BB0_60:
	setp.neu.f32	%p67, %f78, 0f00000000;
	or.pred  	%p68, %p59, %p67;
	@%p68 bra 	BB0_62;
	bra.uni 	BB0_61;

BB0_62:
	mul.f32 	%f191, %f63, %f63;
	fma.rn.f32 	%f192, %f62, %f62, %f191;
	fma.rn.f32 	%f83, %f64, %f64, %f192;
	setp.neu.f32	%p69, %f83, 0f00000000;
	setp.neu.f32	%p70, %f79, 0f00000000;
	or.pred  	%p71, %p69, %p70;
	@%p71 bra 	BB0_64;
	bra.uni 	BB0_63;

BB0_64:
	or.pred  	%p73, %p61, %p59;
	@%p73 bra 	BB0_66;
	bra.uni 	BB0_65;

BB0_66:
	or.pred  	%p76, %p62, %p69;
	@%p76 bra 	BB0_68;
	bra.uni 	BB0_67;

BB0_68:
	sub.f32 	%f210, %f68, %f62;
	sub.f32 	%f211, %f69, %f63;
	sub.f32 	%f212, %f70, %f64;
	sub.f32 	%f213, %f56, %f74;
	mul.f32 	%f214, %f213, 0f3DAAAAAB;
	sub.f32 	%f215, %f57, %f75;
	mul.f32 	%f216, %f215, 0f3DAAAAAB;
	sub.f32 	%f217, %f58, %f76;
	mul.f32 	%f218, %f217, 0f3DAAAAAB;
	fma.rn.f32 	%f262, %f210, 0f3F2AAAAB, %f214;
	fma.rn.f32 	%f263, %f211, 0f3F2AAAAB, %f216;
	fma.rn.f32 	%f264, %f212, 0f3F2AAAAB, %f218;
	bra.uni 	BB0_70;

BB0_61:
	sub.f32 	%f262, %f1, %f62;
	sub.f32 	%f263, %f2, %f63;
	sub.f32 	%f264, %f3, %f64;
	bra.uni 	BB0_70;

BB0_63:
	sub.f32 	%f262, %f68, %f1;
	sub.f32 	%f263, %f69, %f2;
	sub.f32 	%f264, %f70, %f3;
	bra.uni 	BB0_70;

BB0_65:
	mul.f32 	%f193, %f62, 0fC0000000;
	fma.rn.f32 	%f194, %f56, 0f3F000000, %f193;
	add.f32 	%f195, %f63, %f63;
	mul.f32 	%f196, %f57, 0f3F000000;
	sub.f32 	%f197, %f196, %f195;
	add.f32 	%f198, %f64, %f64;
	mul.f32 	%f199, %f58, 0f3F000000;
	sub.f32 	%f200, %f199, %f198;
	fma.rn.f32 	%f262, %f1, 0f3FC00000, %f194;
	fma.rn.f32 	%f263, %f2, 0f3FC00000, %f197;
	fma.rn.f32 	%f264, %f3, 0f3FC00000, %f200;
	bra.uni 	BB0_70;

BB0_67:
	mul.f32 	%f201, %f74, 0fBF000000;
	fma.rn.f32 	%f202, %f68, 0f40000000, %f201;
	mul.f32 	%f203, %f75, 0fBF000000;
	fma.rn.f32 	%f204, %f69, 0f40000000, %f203;
	mul.f32 	%f205, %f76, 0fBF000000;
	fma.rn.f32 	%f206, %f70, 0f40000000, %f205;
	mul.f32 	%f207, %f1, 0f3FC00000;
	sub.f32 	%f262, %f202, %f207;
	mul.f32 	%f208, %f2, 0f3FC00000;
	sub.f32 	%f263, %f204, %f208;
	mul.f32 	%f209, %f3, 0f3FC00000;
	sub.f32 	%f264, %f206, %f209;

BB0_70:
	mul.f32 	%f222, %f249, %f263;
	mul.f32 	%f223, %f248, %f264;
	sub.f32 	%f224, %f223, %f222;
	mul.f32 	%f225, %f247, %f264;
	mul.f32 	%f226, %f249, %f262;
	sub.f32 	%f227, %f226, %f225;
	mul.f32 	%f228, %f248, %f262;
	mul.f32 	%f229, %f247, %f263;
	sub.f32 	%f230, %f229, %f228;
	mul.f32 	%f231, %f2, %f227;
	fma.rn.f32 	%f232, %f1, %f224, %f231;
	fma.rn.f32 	%f233, %f3, %f230, %f232;
	mul.f32 	%f234, %f233, %f102;
	st.global.f32 	[%rd4], %f234;

BB0_72:
	ret;
}


`
)
