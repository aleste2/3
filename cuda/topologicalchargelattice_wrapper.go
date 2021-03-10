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

// CUDA handle for settopologicalchargelattice kernel
var settopologicalchargelattice_code cu.Function

// Stores the arguments for settopologicalchargelattice kernel invocation
type settopologicalchargelattice_args_t struct {
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

// Stores the arguments for settopologicalchargelattice kernel invocation
var settopologicalchargelattice_args settopologicalchargelattice_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	settopologicalchargelattice_args.argptr[0] = unsafe.Pointer(&settopologicalchargelattice_args.arg_s)
	settopologicalchargelattice_args.argptr[1] = unsafe.Pointer(&settopologicalchargelattice_args.arg_mx)
	settopologicalchargelattice_args.argptr[2] = unsafe.Pointer(&settopologicalchargelattice_args.arg_my)
	settopologicalchargelattice_args.argptr[3] = unsafe.Pointer(&settopologicalchargelattice_args.arg_mz)
	settopologicalchargelattice_args.argptr[4] = unsafe.Pointer(&settopologicalchargelattice_args.arg_icxcy)
	settopologicalchargelattice_args.argptr[5] = unsafe.Pointer(&settopologicalchargelattice_args.arg_Nx)
	settopologicalchargelattice_args.argptr[6] = unsafe.Pointer(&settopologicalchargelattice_args.arg_Ny)
	settopologicalchargelattice_args.argptr[7] = unsafe.Pointer(&settopologicalchargelattice_args.arg_Nz)
	settopologicalchargelattice_args.argptr[8] = unsafe.Pointer(&settopologicalchargelattice_args.arg_PBC)
}

// Wrapper for settopologicalchargelattice CUDA kernel, asynchronous.
func k_settopologicalchargelattice_async(s unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, icxcy float32, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("settopologicalchargelattice")
	}

	settopologicalchargelattice_args.Lock()
	defer settopologicalchargelattice_args.Unlock()

	if settopologicalchargelattice_code == 0 {
		settopologicalchargelattice_code = fatbinLoad(settopologicalchargelattice_map, "settopologicalchargelattice")
	}

	settopologicalchargelattice_args.arg_s = s
	settopologicalchargelattice_args.arg_mx = mx
	settopologicalchargelattice_args.arg_my = my
	settopologicalchargelattice_args.arg_mz = mz
	settopologicalchargelattice_args.arg_icxcy = icxcy
	settopologicalchargelattice_args.arg_Nx = Nx
	settopologicalchargelattice_args.arg_Ny = Ny
	settopologicalchargelattice_args.arg_Nz = Nz
	settopologicalchargelattice_args.arg_PBC = PBC

	args := settopologicalchargelattice_args.argptr[:]
	cu.LaunchKernel(settopologicalchargelattice_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("settopologicalchargelattice")
	}
}

// maps compute capability on PTX code for settopologicalchargelattice kernel.
var settopologicalchargelattice_map = map[int]string{0: "",
	70: settopologicalchargelattice_ptx_70}

// settopologicalchargelattice PTX code for various compute capabilities.
const (
	settopologicalchargelattice_ptx_70 = `
.version 7.2
.target sm_70
.address_size 64

	// .globl	settopologicalchargelattice

.visible .entry settopologicalchargelattice(
	.param .u64 settopologicalchargelattice_param_0,
	.param .u64 settopologicalchargelattice_param_1,
	.param .u64 settopologicalchargelattice_param_2,
	.param .u64 settopologicalchargelattice_param_3,
	.param .f32 settopologicalchargelattice_param_4,
	.param .u32 settopologicalchargelattice_param_5,
	.param .u32 settopologicalchargelattice_param_6,
	.param .u32 settopologicalchargelattice_param_7,
	.param .u8 settopologicalchargelattice_param_8
)
{
	.reg .pred 	%p<85>;
	.reg .b16 	%rs<4>;
	.reg .f32 	%f<295>;
	.reg .b32 	%r<157>;
	.reg .b64 	%rd<46>;


	ld.param.u8 	%rs3, [settopologicalchargelattice_param_8];
	ld.param.u64 	%rd5, [settopologicalchargelattice_param_0];
	ld.param.u64 	%rd6, [settopologicalchargelattice_param_1];
	ld.param.u64 	%rd7, [settopologicalchargelattice_param_2];
	ld.param.u64 	%rd8, [settopologicalchargelattice_param_3];
	ld.param.f32 	%f52, [settopologicalchargelattice_param_4];
	ld.param.u32 	%r58, [settopologicalchargelattice_param_5];
	ld.param.u32 	%r59, [settopologicalchargelattice_param_6];
	ld.param.u32 	%r60, [settopologicalchargelattice_param_7];
	cvta.to.global.u64 	%rd1, %rd8;
	cvta.to.global.u64 	%rd2, %rd7;
	cvta.to.global.u64 	%rd3, %rd6;
	mov.u32 	%r61, %ntid.x;
	mov.u32 	%r62, %ctaid.x;
	mov.u32 	%r63, %tid.x;
	mad.lo.s32 	%r1, %r62, %r61, %r63;
	mov.u32 	%r64, %ntid.y;
	mov.u32 	%r65, %ctaid.y;
	mov.u32 	%r66, %tid.y;
	mad.lo.s32 	%r2, %r65, %r64, %r66;
	mov.u32 	%r67, %ntid.z;
	mov.u32 	%r68, %ctaid.z;
	mov.u32 	%r69, %tid.z;
	mad.lo.s32 	%r3, %r68, %r67, %r69;
	setp.ge.s32 	%p3, %r1, %r58;
	setp.ge.s32 	%p4, %r2, %r59;
	or.pred  	%p5, %p3, %p4;
	setp.ge.s32 	%p6, %r3, %r60;
	or.pred  	%p7, %p5, %p6;
	@%p7 bra 	LBB0_72;

	mul.lo.s32 	%r4, %r3, %r59;
	add.s32 	%r70, %r4, %r2;
	mul.lo.s32 	%r5, %r70, %r58;
	add.s32 	%r71, %r5, %r1;
	mul.wide.s32 	%rd9, %r71, 4;
	add.s64 	%rd10, %rd3, %rd9;
	add.s64 	%rd11, %rd2, %rd9;
	add.s64 	%rd12, %rd1, %rd9;
	ld.global.nc.f32 	%f1, [%rd10];
	ld.global.nc.f32 	%f2, [%rd11];
	mul.f32 	%f53, %f2, %f2;
	fma.rn.f32 	%f54, %f1, %f1, %f53;
	ld.global.nc.f32 	%f3, [%rd12];
	fma.rn.f32 	%f55, %f3, %f3, %f54;
	setp.eq.f32 	%p8, %f55, 0f00000000;
	cvta.to.global.u64 	%rd13, %rd5;
	add.s64 	%rd4, %rd13, %rd9;
	@%p8 bra 	LBB0_71;
	bra.uni 	LBB0_2;

LBB0_71:
	mov.u32 	%r144, 0;
	st.global.u32 	[%rd4], %r144;
	bra.uni 	LBB0_72;

LBB0_2:
	and.b16  	%rs1, %rs3, 1;
	setp.eq.s16 	%p9, %rs1, 0;
	add.s32 	%r6, %r1, 1;
	@%p9 bra 	LBB0_4;
	bra.uni 	LBB0_3;

LBB0_4:
	add.s32 	%r74, %r58, -1;
	min.s32 	%r145, %r6, %r74;
	bra.uni 	LBB0_5;

LBB0_3:
	rem.s32 	%r72, %r6, %r58;
	add.s32 	%r73, %r72, %r58;
	rem.s32 	%r145, %r73, %r58;

LBB0_5:
	and.b16  	%rs2, %rs3, 2;
	setp.eq.s16 	%p10, %rs2, 0;
	add.s32 	%r10, %r2, 1;
	@%p10 bra 	LBB0_7;
	bra.uni 	LBB0_6;

LBB0_7:
	add.s32 	%r77, %r59, -1;
	min.s32 	%r146, %r10, %r77;
	bra.uni 	LBB0_8;

LBB0_6:
	rem.s32 	%r75, %r10, %r59;
	add.s32 	%r76, %r75, %r59;
	rem.s32 	%r146, %r76, %r59;

LBB0_8:
	add.s32 	%r14, %r1, -1;
	@%p9 bra 	LBB0_10;
	bra.uni 	LBB0_9;

LBB0_10:
	max.s32 	%r147, %r14, 0;
	bra.uni 	LBB0_11;

LBB0_9:
	rem.s32 	%r78, %r14, %r58;
	add.s32 	%r79, %r78, %r58;
	rem.s32 	%r147, %r79, %r58;

LBB0_11:
	add.s32 	%r18, %r145, %r5;
	add.s32 	%r80, %r146, %r4;
	mad.lo.s32 	%r19, %r80, %r58, %r1;
	add.s32 	%r20, %r147, %r5;
	add.s32 	%r21, %r2, -1;
	@%p10 bra 	LBB0_13;
	bra.uni 	LBB0_12;

LBB0_13:
	max.s32 	%r148, %r21, 0;
	bra.uni 	LBB0_14;

LBB0_12:
	rem.s32 	%r81, %r21, %r59;
	add.s32 	%r82, %r81, %r59;
	rem.s32 	%r148, %r82, %r59;

LBB0_14:
	add.s32 	%r83, %r148, %r4;
	mad.lo.s32 	%r84, %r83, %r58, %r1;
	mul.wide.s32 	%rd14, %r18, 4;
	add.s64 	%rd15, %rd3, %rd14;
	ld.global.nc.f32 	%f4, [%rd15];
	add.s64 	%rd16, %rd2, %rd14;
	ld.global.nc.f32 	%f5, [%rd16];
	add.s64 	%rd17, %rd1, %rd14;
	ld.global.nc.f32 	%f6, [%rd17];
	mul.wide.s32 	%rd18, %r19, 4;
	add.s64 	%rd19, %rd3, %rd18;
	ld.global.nc.f32 	%f7, [%rd19];
	add.s64 	%rd20, %rd2, %rd18;
	ld.global.nc.f32 	%f8, [%rd20];
	add.s64 	%rd21, %rd1, %rd18;
	ld.global.nc.f32 	%f9, [%rd21];
	mul.wide.s32 	%rd22, %r20, 4;
	add.s64 	%rd23, %rd3, %rd22;
	ld.global.nc.f32 	%f10, [%rd23];
	add.s64 	%rd24, %rd2, %rd22;
	ld.global.nc.f32 	%f11, [%rd24];
	add.s64 	%rd25, %rd1, %rd22;
	ld.global.nc.f32 	%f12, [%rd25];
	mul.wide.s32 	%rd26, %r84, 4;
	add.s64 	%rd27, %rd3, %rd26;
	ld.global.nc.f32 	%f13, [%rd27];
	add.s64 	%rd28, %rd2, %rd26;
	ld.global.nc.f32 	%f14, [%rd28];
	add.s64 	%rd29, %rd1, %rd26;
	ld.global.nc.f32 	%f15, [%rd29];
	setp.ne.s16 	%p13, %rs1, 0;
	setp.lt.s32 	%p14, %r6, %r58;
	or.pred  	%p1, %p14, %p13;
	mov.f32 	%f290, 0f00000000;
	not.pred 	%p15, %p1;
	@%p15 bra 	LBB0_28;

	setp.ge.s32 	%p16, %r10, %r59;
	and.pred  	%p18, %p16, %p10;
	@%p18 bra 	LBB0_28;

	@%p10 bra 	LBB0_18;
	bra.uni 	LBB0_17;

LBB0_18:
	add.s32 	%r87, %r59, -1;
	min.s32 	%r149, %r10, %r87;
	bra.uni 	LBB0_19;

LBB0_17:
	rem.s32 	%r85, %r10, %r59;
	add.s32 	%r86, %r85, %r59;
	rem.s32 	%r149, %r86, %r59;

LBB0_19:
	@%p9 bra 	LBB0_21;
	bra.uni 	LBB0_20;

LBB0_21:
	add.s32 	%r90, %r58, -1;
	min.s32 	%r150, %r6, %r90;
	bra.uni 	LBB0_22;

LBB0_20:
	rem.s32 	%r88, %r6, %r58;
	add.s32 	%r89, %r88, %r58;
	rem.s32 	%r150, %r89, %r58;

LBB0_22:
	add.s32 	%r91, %r149, %r4;
	mad.lo.s32 	%r92, %r91, %r58, %r150;
	mul.wide.s32 	%rd30, %r92, 4;
	add.s64 	%rd31, %rd3, %rd30;
	add.s64 	%rd32, %rd2, %rd30;
	add.s64 	%rd33, %rd1, %rd30;
	ld.global.nc.f32 	%f58, [%rd31];
	ld.global.nc.f32 	%f59, [%rd32];
	mul.f32 	%f60, %f59, %f59;
	fma.rn.f32 	%f61, %f58, %f58, %f60;
	ld.global.nc.f32 	%f62, [%rd33];
	fma.rn.f32 	%f16, %f62, %f62, %f61;
	mul.f32 	%f63, %f6, %f8;
	mul.f32 	%f64, %f5, %f9;
	sub.f32 	%f65, %f64, %f63;
	mul.f32 	%f66, %f4, %f9;
	mul.f32 	%f67, %f6, %f7;
	sub.f32 	%f68, %f67, %f66;
	mul.f32 	%f69, %f5, %f7;
	mul.f32 	%f70, %f4, %f8;
	sub.f32 	%f71, %f70, %f69;
	mul.f32 	%f72, %f2, %f68;
	fma.rn.f32 	%f73, %f1, %f65, %f72;
	fma.rn.f32 	%f74, %f3, %f71, %f73;
	mul.f32 	%f75, %f2, %f5;
	fma.rn.f32 	%f76, %f1, %f4, %f75;
	fma.rn.f32 	%f77, %f3, %f6, %f76;
	add.f32 	%f78, %f77, 0f3F800000;
	mul.f32 	%f79, %f2, %f8;
	fma.rn.f32 	%f80, %f1, %f7, %f79;
	fma.rn.f32 	%f81, %f3, %f9, %f80;
	add.f32 	%f82, %f78, %f81;
	mul.f32 	%f83, %f5, %f8;
	fma.rn.f32 	%f84, %f4, %f7, %f83;
	fma.rn.f32 	%f85, %f6, %f9, %f84;
	add.f32 	%f86, %f85, %f82;
	abs.f32 	%f17, %f86;
	abs.f32 	%f18, %f74;
	setp.eq.f32 	%p21, %f17, 0f00000000;
	setp.eq.f32 	%p22, %f18, 0f00000000;
	and.pred  	%p23, %p21, %p22;
	mov.b32 	%r31, %f86;
	mov.b32 	%r93, %f74;
	and.b32  	%r32, %r93, -2147483648;
	@%p23 bra 	LBB0_26;
	bra.uni 	LBB0_23;

LBB0_26:
	shr.s32 	%r98, %r31, 31;
	and.b32  	%r99, %r98, 1078530011;
	or.b32  	%r100, %r99, %r32;
	mov.b32 	%f287, %r100;
	bra.uni 	LBB0_27;

LBB0_23:
	setp.eq.f32 	%p24, %f17, 0f7F800000;
	setp.eq.f32 	%p25, %f18, 0f7F800000;
	and.pred  	%p26, %p24, %p25;
	@%p26 bra 	LBB0_25;
	bra.uni 	LBB0_24;

LBB0_25:
	setp.lt.s32 	%p30, %r31, 0;
	selp.b32 	%r96, 1075235812, 1061752795, %p30;
	or.b32  	%r97, %r96, %r32;
	mov.b32 	%f287, %r97;
	bra.uni 	LBB0_27;

LBB0_24:
	setp.lt.s32 	%p27, %r31, 0;
	min.f32 	%f87, %f18, %f17;
	max.f32 	%f88, %f18, %f17;
	div.rn.f32 	%f89, %f87, %f88;
	mul.rn.f32 	%f90, %f89, %f89;
	mov.f32 	%f91, 0fC0B59883;
	mov.f32 	%f92, 0fBF52C7EA;
	fma.rn.f32 	%f93, %f90, %f92, %f91;
	mov.f32 	%f94, 0fC0D21907;
	fma.rn.f32 	%f95, %f93, %f90, %f94;
	mul.f32 	%f96, %f90, %f95;
	mul.f32 	%f97, %f89, %f96;
	add.f32 	%f98, %f90, 0f41355DC0;
	mov.f32 	%f99, 0f41E6BD60;
	fma.rn.f32 	%f100, %f98, %f90, %f99;
	mov.f32 	%f101, 0f419D92C8;
	fma.rn.f32 	%f102, %f100, %f90, %f101;
	rcp.rn.f32 	%f103, %f102;
	fma.rn.f32 	%f104, %f97, %f103, %f89;
	mov.f32 	%f105, 0f3FC90FDB;
	sub.f32 	%f106, %f105, %f104;
	setp.gt.f32 	%p28, %f18, %f17;
	selp.f32 	%f107, %f106, %f104, %p28;
	mov.f32 	%f108, 0f40490FDB;
	sub.f32 	%f109, %f108, %f107;
	selp.f32 	%f110, %f109, %f107, %p27;
	mov.b32 	%r94, %f110;
	or.b32  	%r95, %r32, %r94;
	mov.b32 	%f111, %r95;
	add.f32 	%f112, %f17, %f18;
	setp.le.f32 	%p29, %f112, 0f7F800000;
	selp.f32 	%f287, %f111, %f112, %p29;

LBB0_27:
	add.f32 	%f113, %f287, %f287;
	setp.eq.f32 	%p31, %f16, 0f00000000;
	selp.f32 	%f114, 0f3F800000, 0f3F000000, %p31;
	fma.rn.f32 	%f290, %f114, %f113, 0f00000000;

LBB0_28:
	setp.gt.s32 	%p32, %r1, 0;
	or.pred  	%p2, %p32, %p13;
	not.pred 	%p34, %p2;
	@%p34 bra 	LBB0_42;

	setp.ge.s32 	%p35, %r10, %r59;
	and.pred  	%p37, %p35, %p10;
	@%p37 bra 	LBB0_42;

	@%p10 bra 	LBB0_32;
	bra.uni 	LBB0_31;

LBB0_32:
	add.s32 	%r103, %r59, -1;
	min.s32 	%r151, %r10, %r103;
	bra.uni 	LBB0_33;

LBB0_31:
	rem.s32 	%r101, %r10, %r59;
	add.s32 	%r102, %r101, %r59;
	rem.s32 	%r151, %r102, %r59;

LBB0_33:
	@%p9 bra 	LBB0_35;
	bra.uni 	LBB0_34;

LBB0_35:
	max.s32 	%r152, %r14, 0;
	bra.uni 	LBB0_36;

LBB0_34:
	rem.s32 	%r104, %r14, %r58;
	add.s32 	%r105, %r104, %r58;
	rem.s32 	%r152, %r105, %r58;

LBB0_36:
	add.s32 	%r106, %r151, %r4;
	mad.lo.s32 	%r107, %r106, %r58, %r152;
	mul.wide.s32 	%rd34, %r107, 4;
	add.s64 	%rd35, %rd3, %rd34;
	add.s64 	%rd36, %rd2, %rd34;
	add.s64 	%rd37, %rd1, %rd34;
	ld.global.nc.f32 	%f115, [%rd35];
	ld.global.nc.f32 	%f116, [%rd36];
	mul.f32 	%f117, %f116, %f116;
	fma.rn.f32 	%f118, %f115, %f115, %f117;
	ld.global.nc.f32 	%f119, [%rd37];
	fma.rn.f32 	%f25, %f119, %f119, %f118;
	mul.f32 	%f120, %f9, %f11;
	mul.f32 	%f121, %f8, %f12;
	sub.f32 	%f122, %f121, %f120;
	mul.f32 	%f123, %f7, %f12;
	mul.f32 	%f124, %f9, %f10;
	sub.f32 	%f125, %f124, %f123;
	mul.f32 	%f126, %f8, %f10;
	mul.f32 	%f127, %f7, %f11;
	sub.f32 	%f128, %f127, %f126;
	mul.f32 	%f129, %f2, %f125;
	fma.rn.f32 	%f130, %f1, %f122, %f129;
	fma.rn.f32 	%f131, %f3, %f128, %f130;
	mul.f32 	%f132, %f2, %f8;
	fma.rn.f32 	%f133, %f1, %f7, %f132;
	fma.rn.f32 	%f134, %f3, %f9, %f133;
	add.f32 	%f135, %f134, 0f3F800000;
	mul.f32 	%f136, %f2, %f11;
	fma.rn.f32 	%f137, %f1, %f10, %f136;
	fma.rn.f32 	%f138, %f3, %f12, %f137;
	add.f32 	%f139, %f135, %f138;
	mul.f32 	%f140, %f8, %f11;
	fma.rn.f32 	%f141, %f7, %f10, %f140;
	fma.rn.f32 	%f142, %f9, %f12, %f141;
	add.f32 	%f143, %f142, %f139;
	abs.f32 	%f26, %f143;
	abs.f32 	%f27, %f131;
	setp.eq.f32 	%p40, %f26, 0f00000000;
	setp.eq.f32 	%p41, %f27, 0f00000000;
	and.pred  	%p42, %p40, %p41;
	mov.b32 	%r39, %f143;
	mov.b32 	%r108, %f131;
	and.b32  	%r40, %r108, -2147483648;
	@%p42 bra 	LBB0_40;
	bra.uni 	LBB0_37;

LBB0_40:
	shr.s32 	%r113, %r39, 31;
	and.b32  	%r114, %r113, 1078530011;
	or.b32  	%r115, %r114, %r40;
	mov.b32 	%f289, %r115;
	bra.uni 	LBB0_41;

LBB0_37:
	setp.eq.f32 	%p43, %f26, 0f7F800000;
	setp.eq.f32 	%p44, %f27, 0f7F800000;
	and.pred  	%p45, %p43, %p44;
	@%p45 bra 	LBB0_39;
	bra.uni 	LBB0_38;

LBB0_39:
	setp.lt.s32 	%p49, %r39, 0;
	selp.b32 	%r111, 1075235812, 1061752795, %p49;
	or.b32  	%r112, %r111, %r40;
	mov.b32 	%f289, %r112;
	bra.uni 	LBB0_41;

LBB0_38:
	setp.lt.s32 	%p46, %r39, 0;
	min.f32 	%f144, %f27, %f26;
	max.f32 	%f145, %f27, %f26;
	div.rn.f32 	%f146, %f144, %f145;
	mul.rn.f32 	%f147, %f146, %f146;
	mov.f32 	%f148, 0fC0B59883;
	mov.f32 	%f149, 0fBF52C7EA;
	fma.rn.f32 	%f150, %f147, %f149, %f148;
	mov.f32 	%f151, 0fC0D21907;
	fma.rn.f32 	%f152, %f150, %f147, %f151;
	mul.f32 	%f153, %f147, %f152;
	mul.f32 	%f154, %f146, %f153;
	add.f32 	%f155, %f147, 0f41355DC0;
	mov.f32 	%f156, 0f41E6BD60;
	fma.rn.f32 	%f157, %f155, %f147, %f156;
	mov.f32 	%f158, 0f419D92C8;
	fma.rn.f32 	%f159, %f157, %f147, %f158;
	rcp.rn.f32 	%f160, %f159;
	fma.rn.f32 	%f161, %f154, %f160, %f146;
	mov.f32 	%f162, 0f3FC90FDB;
	sub.f32 	%f163, %f162, %f161;
	setp.gt.f32 	%p47, %f27, %f26;
	selp.f32 	%f164, %f163, %f161, %p47;
	mov.f32 	%f165, 0f40490FDB;
	sub.f32 	%f166, %f165, %f164;
	selp.f32 	%f167, %f166, %f164, %p46;
	mov.b32 	%r109, %f167;
	or.b32  	%r110, %r40, %r109;
	mov.b32 	%f168, %r110;
	add.f32 	%f169, %f26, %f27;
	setp.le.f32 	%p48, %f169, 0f7F800000;
	selp.f32 	%f289, %f168, %f169, %p48;

LBB0_41:
	add.f32 	%f170, %f289, %f289;
	setp.eq.f32 	%p50, %f25, 0f00000000;
	selp.f32 	%f171, 0f3F800000, 0f3F000000, %p50;
	fma.rn.f32 	%f290, %f171, %f170, %f290;

LBB0_42:
	@%p34 bra 	LBB0_56;

	setp.lt.s32 	%p52, %r2, 1;
	and.pred  	%p54, %p52, %p10;
	@%p54 bra 	LBB0_56;

	@%p10 bra 	LBB0_46;
	bra.uni 	LBB0_45;

LBB0_46:
	max.s32 	%r153, %r21, 0;
	bra.uni 	LBB0_47;

LBB0_45:
	rem.s32 	%r116, %r21, %r59;
	add.s32 	%r117, %r116, %r59;
	rem.s32 	%r153, %r117, %r59;

LBB0_47:
	@%p9 bra 	LBB0_49;
	bra.uni 	LBB0_48;

LBB0_49:
	max.s32 	%r154, %r14, 0;
	bra.uni 	LBB0_50;

LBB0_48:
	rem.s32 	%r118, %r14, %r58;
	add.s32 	%r119, %r118, %r58;
	rem.s32 	%r154, %r119, %r58;

LBB0_50:
	add.s32 	%r120, %r153, %r4;
	mad.lo.s32 	%r121, %r120, %r58, %r154;
	mul.wide.s32 	%rd38, %r121, 4;
	add.s64 	%rd39, %rd3, %rd38;
	add.s64 	%rd40, %rd2, %rd38;
	add.s64 	%rd41, %rd1, %rd38;
	ld.global.nc.f32 	%f172, [%rd39];
	ld.global.nc.f32 	%f173, [%rd40];
	mul.f32 	%f174, %f173, %f173;
	fma.rn.f32 	%f175, %f172, %f172, %f174;
	ld.global.nc.f32 	%f176, [%rd41];
	fma.rn.f32 	%f34, %f176, %f176, %f175;
	mul.f32 	%f177, %f12, %f14;
	mul.f32 	%f178, %f11, %f15;
	sub.f32 	%f179, %f178, %f177;
	mul.f32 	%f180, %f10, %f15;
	mul.f32 	%f181, %f12, %f13;
	sub.f32 	%f182, %f181, %f180;
	mul.f32 	%f183, %f11, %f13;
	mul.f32 	%f184, %f10, %f14;
	sub.f32 	%f185, %f184, %f183;
	mul.f32 	%f186, %f2, %f182;
	fma.rn.f32 	%f187, %f1, %f179, %f186;
	fma.rn.f32 	%f188, %f3, %f185, %f187;
	mul.f32 	%f189, %f2, %f11;
	fma.rn.f32 	%f190, %f1, %f10, %f189;
	fma.rn.f32 	%f191, %f3, %f12, %f190;
	add.f32 	%f192, %f191, 0f3F800000;
	mul.f32 	%f193, %f2, %f14;
	fma.rn.f32 	%f194, %f1, %f13, %f193;
	fma.rn.f32 	%f195, %f3, %f15, %f194;
	add.f32 	%f196, %f192, %f195;
	mul.f32 	%f197, %f11, %f14;
	fma.rn.f32 	%f198, %f10, %f13, %f197;
	fma.rn.f32 	%f199, %f12, %f15, %f198;
	add.f32 	%f200, %f199, %f196;
	abs.f32 	%f35, %f200;
	abs.f32 	%f36, %f188;
	setp.eq.f32 	%p57, %f35, 0f00000000;
	setp.eq.f32 	%p58, %f36, 0f00000000;
	and.pred  	%p59, %p57, %p58;
	mov.b32 	%r47, %f200;
	mov.b32 	%r122, %f188;
	and.b32  	%r48, %r122, -2147483648;
	@%p59 bra 	LBB0_54;
	bra.uni 	LBB0_51;

LBB0_54:
	shr.s32 	%r127, %r47, 31;
	and.b32  	%r128, %r127, 1078530011;
	or.b32  	%r129, %r128, %r48;
	mov.b32 	%f291, %r129;
	bra.uni 	LBB0_55;

LBB0_51:
	setp.eq.f32 	%p60, %f35, 0f7F800000;
	setp.eq.f32 	%p61, %f36, 0f7F800000;
	and.pred  	%p62, %p60, %p61;
	@%p62 bra 	LBB0_53;
	bra.uni 	LBB0_52;

LBB0_53:
	setp.lt.s32 	%p66, %r47, 0;
	selp.b32 	%r125, 1075235812, 1061752795, %p66;
	or.b32  	%r126, %r125, %r48;
	mov.b32 	%f291, %r126;
	bra.uni 	LBB0_55;

LBB0_52:
	setp.lt.s32 	%p63, %r47, 0;
	min.f32 	%f201, %f36, %f35;
	max.f32 	%f202, %f36, %f35;
	div.rn.f32 	%f203, %f201, %f202;
	mul.rn.f32 	%f204, %f203, %f203;
	mov.f32 	%f205, 0fC0B59883;
	mov.f32 	%f206, 0fBF52C7EA;
	fma.rn.f32 	%f207, %f204, %f206, %f205;
	mov.f32 	%f208, 0fC0D21907;
	fma.rn.f32 	%f209, %f207, %f204, %f208;
	mul.f32 	%f210, %f204, %f209;
	mul.f32 	%f211, %f203, %f210;
	add.f32 	%f212, %f204, 0f41355DC0;
	mov.f32 	%f213, 0f41E6BD60;
	fma.rn.f32 	%f214, %f212, %f204, %f213;
	mov.f32 	%f215, 0f419D92C8;
	fma.rn.f32 	%f216, %f214, %f204, %f215;
	rcp.rn.f32 	%f217, %f216;
	fma.rn.f32 	%f218, %f211, %f217, %f203;
	mov.f32 	%f219, 0f3FC90FDB;
	sub.f32 	%f220, %f219, %f218;
	setp.gt.f32 	%p64, %f36, %f35;
	selp.f32 	%f221, %f220, %f218, %p64;
	mov.f32 	%f222, 0f40490FDB;
	sub.f32 	%f223, %f222, %f221;
	selp.f32 	%f224, %f223, %f221, %p63;
	mov.b32 	%r123, %f224;
	or.b32  	%r124, %r48, %r123;
	mov.b32 	%f225, %r124;
	add.f32 	%f226, %f35, %f36;
	setp.le.f32 	%p65, %f226, 0f7F800000;
	selp.f32 	%f291, %f225, %f226, %p65;

LBB0_55:
	add.f32 	%f227, %f291, %f291;
	setp.eq.f32 	%p67, %f34, 0f00000000;
	selp.f32 	%f228, 0f3F800000, 0f3F000000, %p67;
	fma.rn.f32 	%f290, %f228, %f227, %f290;

LBB0_56:
	@%p15 bra 	LBB0_70;

	setp.lt.s32 	%p69, %r2, 1;
	and.pred  	%p71, %p69, %p10;
	@%p71 bra 	LBB0_70;

	@%p10 bra 	LBB0_60;
	bra.uni 	LBB0_59;

LBB0_60:
	max.s32 	%r155, %r21, 0;
	bra.uni 	LBB0_61;

LBB0_59:
	rem.s32 	%r130, %r21, %r59;
	add.s32 	%r131, %r130, %r59;
	rem.s32 	%r155, %r131, %r59;

LBB0_61:
	add.s32 	%r52, %r155, %r4;
	@%p9 bra 	LBB0_63;
	bra.uni 	LBB0_62;

LBB0_63:
	add.s32 	%r134, %r58, -1;
	min.s32 	%r156, %r6, %r134;
	bra.uni 	LBB0_64;

LBB0_62:
	rem.s32 	%r132, %r6, %r58;
	add.s32 	%r133, %r132, %r58;
	rem.s32 	%r156, %r133, %r58;

LBB0_64:
	mad.lo.s32 	%r135, %r52, %r58, %r156;
	mul.wide.s32 	%rd42, %r135, 4;
	add.s64 	%rd43, %rd3, %rd42;
	add.s64 	%rd44, %rd2, %rd42;
	add.s64 	%rd45, %rd1, %rd42;
	ld.global.nc.f32 	%f229, [%rd43];
	ld.global.nc.f32 	%f230, [%rd44];
	mul.f32 	%f231, %f230, %f230;
	fma.rn.f32 	%f232, %f229, %f229, %f231;
	ld.global.nc.f32 	%f233, [%rd45];
	fma.rn.f32 	%f43, %f233, %f233, %f232;
	mul.f32 	%f234, %f5, %f15;
	mul.f32 	%f235, %f6, %f14;
	sub.f32 	%f236, %f235, %f234;
	mul.f32 	%f237, %f6, %f13;
	mul.f32 	%f238, %f4, %f15;
	sub.f32 	%f239, %f238, %f237;
	mul.f32 	%f240, %f4, %f14;
	mul.f32 	%f241, %f5, %f13;
	sub.f32 	%f242, %f241, %f240;
	mul.f32 	%f243, %f2, %f239;
	fma.rn.f32 	%f244, %f1, %f236, %f243;
	fma.rn.f32 	%f245, %f3, %f242, %f244;
	mul.f32 	%f246, %f2, %f14;
	fma.rn.f32 	%f247, %f1, %f13, %f246;
	fma.rn.f32 	%f248, %f3, %f15, %f247;
	add.f32 	%f249, %f248, 0f3F800000;
	mul.f32 	%f250, %f2, %f5;
	fma.rn.f32 	%f251, %f1, %f4, %f250;
	fma.rn.f32 	%f252, %f3, %f6, %f251;
	add.f32 	%f253, %f252, %f249;
	mul.f32 	%f254, %f5, %f14;
	fma.rn.f32 	%f255, %f4, %f13, %f254;
	fma.rn.f32 	%f256, %f6, %f15, %f255;
	add.f32 	%f257, %f256, %f253;
	abs.f32 	%f44, %f257;
	abs.f32 	%f45, %f245;
	setp.eq.f32 	%p74, %f44, 0f00000000;
	setp.eq.f32 	%p75, %f45, 0f00000000;
	and.pred  	%p76, %p74, %p75;
	mov.b32 	%r56, %f257;
	mov.b32 	%r136, %f245;
	and.b32  	%r57, %r136, -2147483648;
	@%p76 bra 	LBB0_68;
	bra.uni 	LBB0_65;

LBB0_68:
	shr.s32 	%r141, %r56, 31;
	and.b32  	%r142, %r141, 1078530011;
	or.b32  	%r143, %r142, %r57;
	mov.b32 	%f293, %r143;
	bra.uni 	LBB0_69;

LBB0_65:
	setp.eq.f32 	%p77, %f44, 0f7F800000;
	setp.eq.f32 	%p78, %f45, 0f7F800000;
	and.pred  	%p79, %p77, %p78;
	@%p79 bra 	LBB0_67;
	bra.uni 	LBB0_66;

LBB0_67:
	setp.lt.s32 	%p83, %r56, 0;
	selp.b32 	%r139, 1075235812, 1061752795, %p83;
	or.b32  	%r140, %r139, %r57;
	mov.b32 	%f293, %r140;
	bra.uni 	LBB0_69;

LBB0_66:
	setp.lt.s32 	%p80, %r56, 0;
	min.f32 	%f258, %f45, %f44;
	max.f32 	%f259, %f45, %f44;
	div.rn.f32 	%f260, %f258, %f259;
	mul.rn.f32 	%f261, %f260, %f260;
	mov.f32 	%f262, 0fC0B59883;
	mov.f32 	%f263, 0fBF52C7EA;
	fma.rn.f32 	%f264, %f261, %f263, %f262;
	mov.f32 	%f265, 0fC0D21907;
	fma.rn.f32 	%f266, %f264, %f261, %f265;
	mul.f32 	%f267, %f261, %f266;
	mul.f32 	%f268, %f260, %f267;
	add.f32 	%f269, %f261, 0f41355DC0;
	mov.f32 	%f270, 0f41E6BD60;
	fma.rn.f32 	%f271, %f269, %f261, %f270;
	mov.f32 	%f272, 0f419D92C8;
	fma.rn.f32 	%f273, %f271, %f261, %f272;
	rcp.rn.f32 	%f274, %f273;
	fma.rn.f32 	%f275, %f268, %f274, %f260;
	mov.f32 	%f276, 0f3FC90FDB;
	sub.f32 	%f277, %f276, %f275;
	setp.gt.f32 	%p81, %f45, %f44;
	selp.f32 	%f278, %f277, %f275, %p81;
	mov.f32 	%f279, 0f40490FDB;
	sub.f32 	%f280, %f279, %f278;
	selp.f32 	%f281, %f280, %f278, %p80;
	mov.b32 	%r137, %f281;
	or.b32  	%r138, %r57, %r137;
	mov.b32 	%f282, %r138;
	add.f32 	%f283, %f44, %f45;
	setp.le.f32 	%p82, %f283, 0f7F800000;
	selp.f32 	%f293, %f282, %f283, %p82;

LBB0_69:
	add.f32 	%f284, %f293, %f293;
	setp.eq.f32 	%p84, %f43, 0f00000000;
	selp.f32 	%f285, 0f3F800000, 0f3F000000, %p84;
	fma.rn.f32 	%f290, %f285, %f284, %f290;

LBB0_70:
	mul.f32 	%f286, %f290, %f52;
	st.global.f32 	[%rd4], %f286;

LBB0_72:
	ret;

}

`
)
