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

// CUDA handle for getmagnetoelasticforce kernel
var getmagnetoelasticforce_code cu.Function

// Stores the arguments for getmagnetoelasticforce kernel invocation
type getmagnetoelasticforce_args_t struct {
	arg_fx     unsafe.Pointer
	arg_fy     unsafe.Pointer
	arg_fz     unsafe.Pointer
	arg_mx     unsafe.Pointer
	arg_my     unsafe.Pointer
	arg_mz     unsafe.Pointer
	arg_B1_    unsafe.Pointer
	arg_B1_mul float32
	arg_B2_    unsafe.Pointer
	arg_B2_mul float32
	arg_rcsx   float32
	arg_rcsy   float32
	arg_rcsz   float32
	arg_Nx     int
	arg_Ny     int
	arg_Nz     int
	arg_PBC    byte
	argptr     [17]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for getmagnetoelasticforce kernel invocation
var getmagnetoelasticforce_args getmagnetoelasticforce_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	getmagnetoelasticforce_args.argptr[0] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_fx)
	getmagnetoelasticforce_args.argptr[1] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_fy)
	getmagnetoelasticforce_args.argptr[2] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_fz)
	getmagnetoelasticforce_args.argptr[3] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_mx)
	getmagnetoelasticforce_args.argptr[4] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_my)
	getmagnetoelasticforce_args.argptr[5] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_mz)
	getmagnetoelasticforce_args.argptr[6] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_B1_)
	getmagnetoelasticforce_args.argptr[7] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_B1_mul)
	getmagnetoelasticforce_args.argptr[8] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_B2_)
	getmagnetoelasticforce_args.argptr[9] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_B2_mul)
	getmagnetoelasticforce_args.argptr[10] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_rcsx)
	getmagnetoelasticforce_args.argptr[11] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_rcsy)
	getmagnetoelasticforce_args.argptr[12] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_rcsz)
	getmagnetoelasticforce_args.argptr[13] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_Nx)
	getmagnetoelasticforce_args.argptr[14] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_Ny)
	getmagnetoelasticforce_args.argptr[15] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_Nz)
	getmagnetoelasticforce_args.argptr[16] = unsafe.Pointer(&getmagnetoelasticforce_args.arg_PBC)
}

// Wrapper for getmagnetoelasticforce CUDA kernel, asynchronous.
func k_getmagnetoelasticforce_async(fx unsafe.Pointer, fy unsafe.Pointer, fz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, B1_ unsafe.Pointer, B1_mul float32, B2_ unsafe.Pointer, B2_mul float32, rcsx float32, rcsy float32, rcsz float32, Nx int, Ny int, Nz int, PBC byte, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("getmagnetoelasticforce")
	}

	getmagnetoelasticforce_args.Lock()
	defer getmagnetoelasticforce_args.Unlock()

	if getmagnetoelasticforce_code == 0 {
		getmagnetoelasticforce_code = fatbinLoad(getmagnetoelasticforce_map, "getmagnetoelasticforce")
	}

	getmagnetoelasticforce_args.arg_fx = fx
	getmagnetoelasticforce_args.arg_fy = fy
	getmagnetoelasticforce_args.arg_fz = fz
	getmagnetoelasticforce_args.arg_mx = mx
	getmagnetoelasticforce_args.arg_my = my
	getmagnetoelasticforce_args.arg_mz = mz
	getmagnetoelasticforce_args.arg_B1_ = B1_
	getmagnetoelasticforce_args.arg_B1_mul = B1_mul
	getmagnetoelasticforce_args.arg_B2_ = B2_
	getmagnetoelasticforce_args.arg_B2_mul = B2_mul
	getmagnetoelasticforce_args.arg_rcsx = rcsx
	getmagnetoelasticforce_args.arg_rcsy = rcsy
	getmagnetoelasticforce_args.arg_rcsz = rcsz
	getmagnetoelasticforce_args.arg_Nx = Nx
	getmagnetoelasticforce_args.arg_Ny = Ny
	getmagnetoelasticforce_args.arg_Nz = Nz
	getmagnetoelasticforce_args.arg_PBC = PBC

	args := getmagnetoelasticforce_args.argptr[:]
	cu.LaunchKernel(getmagnetoelasticforce_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("getmagnetoelasticforce")
	}
}

// maps compute capability on PTX code for getmagnetoelasticforce kernel.
var getmagnetoelasticforce_map = map[int]string{0: "",
	70: getmagnetoelasticforce_ptx_70}

// getmagnetoelasticforce PTX code for various compute capabilities.
const (
	getmagnetoelasticforce_ptx_70 = `
.version 7.2
.target sm_70
.address_size 64

	// .globl	getmagnetoelasticforce

.visible .entry getmagnetoelasticforce(
	.param .u64 getmagnetoelasticforce_param_0,
	.param .u64 getmagnetoelasticforce_param_1,
	.param .u64 getmagnetoelasticforce_param_2,
	.param .u64 getmagnetoelasticforce_param_3,
	.param .u64 getmagnetoelasticforce_param_4,
	.param .u64 getmagnetoelasticforce_param_5,
	.param .u64 getmagnetoelasticforce_param_6,
	.param .f32 getmagnetoelasticforce_param_7,
	.param .u64 getmagnetoelasticforce_param_8,
	.param .f32 getmagnetoelasticforce_param_9,
	.param .f32 getmagnetoelasticforce_param_10,
	.param .f32 getmagnetoelasticforce_param_11,
	.param .f32 getmagnetoelasticforce_param_12,
	.param .u32 getmagnetoelasticforce_param_13,
	.param .u32 getmagnetoelasticforce_param_14,
	.param .u32 getmagnetoelasticforce_param_15,
	.param .u8 getmagnetoelasticforce_param_16
)
{
	.reg .pred 	%p<116>;
	.reg .b16 	%rs<5>;
	.reg .f32 	%f<462>;
	.reg .b32 	%r<130>;
	.reg .b64 	%rd<77>;


	ld.param.u8 	%rs4, [getmagnetoelasticforce_param_16];
	ld.param.u64 	%rd4, [getmagnetoelasticforce_param_0];
	ld.param.u64 	%rd5, [getmagnetoelasticforce_param_1];
	ld.param.u64 	%rd6, [getmagnetoelasticforce_param_2];
	ld.param.u64 	%rd9, [getmagnetoelasticforce_param_3];
	ld.param.u64 	%rd10, [getmagnetoelasticforce_param_4];
	ld.param.u64 	%rd11, [getmagnetoelasticforce_param_5];
	ld.param.u64 	%rd7, [getmagnetoelasticforce_param_6];
	ld.param.f32 	%f460, [getmagnetoelasticforce_param_7];
	ld.param.u64 	%rd8, [getmagnetoelasticforce_param_8];
	ld.param.f32 	%f461, [getmagnetoelasticforce_param_9];
	ld.param.f32 	%f193, [getmagnetoelasticforce_param_10];
	ld.param.f32 	%f194, [getmagnetoelasticforce_param_11];
	ld.param.f32 	%f195, [getmagnetoelasticforce_param_12];
	ld.param.u32 	%r58, [getmagnetoelasticforce_param_13];
	ld.param.u32 	%r59, [getmagnetoelasticforce_param_14];
	ld.param.u32 	%r60, [getmagnetoelasticforce_param_15];
	cvta.to.global.u64 	%rd1, %rd11;
	cvta.to.global.u64 	%rd2, %rd10;
	cvta.to.global.u64 	%rd3, %rd9;
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
	setp.ge.s32 	%p1, %r1, %r58;
	setp.ge.s32 	%p2, %r2, %r59;
	or.pred  	%p3, %p1, %p2;
	setp.ge.s32 	%p4, %r3, %r60;
	or.pred  	%p5, %p3, %p4;
	@%p5 bra 	LBB0_108;

	mul.lo.s32 	%r4, %r3, %r59;
	add.s32 	%r70, %r4, %r2;
	mul.lo.s32 	%r5, %r70, %r58;
	add.s32 	%r6, %r5, %r1;
	mul.wide.s32 	%rd12, %r6, 4;
	add.s64 	%rd13, %rd3, %rd12;
	ld.global.nc.f32 	%f1, [%rd13];
	add.s64 	%rd14, %rd2, %rd12;
	ld.global.nc.f32 	%f2, [%rd14];
	add.s64 	%rd15, %rd1, %rd12;
	ld.global.nc.f32 	%f3, [%rd15];
	and.b16  	%rs1, %rs4, 1;
	setp.eq.s16 	%p6, %rs1, 0;
	add.s32 	%r7, %r1, -2;
	@%p6 bra 	LBB0_3;
	bra.uni 	LBB0_2;

LBB0_3:
	max.s32 	%r118, %r7, 0;
	bra.uni 	LBB0_4;

LBB0_2:
	rem.s32 	%r71, %r7, %r58;
	add.s32 	%r72, %r71, %r58;
	rem.s32 	%r118, %r72, %r58;

LBB0_4:
	setp.lt.s32 	%p8, %r1, 2;
	mov.f32 	%f7, 0f00000000;
	and.pred  	%p9, %p8, %p6;
	mov.f32 	%f8, %f7;
	mov.f32 	%f9, %f7;
	@%p9 bra 	LBB0_6;

	add.s32 	%r73, %r118, %r5;
	mul.wide.s32 	%rd16, %r73, 4;
	add.s64 	%rd17, %rd3, %rd16;
	add.s64 	%rd18, %rd2, %rd16;
	add.s64 	%rd19, %rd1, %rd16;
	ld.global.nc.f32 	%f9, [%rd19];
	ld.global.nc.f32 	%f8, [%rd18];
	ld.global.nc.f32 	%f7, [%rd17];

LBB0_6:
	add.s32 	%r11, %r1, -1;
	@%p6 bra 	LBB0_8;
	bra.uni 	LBB0_7;

LBB0_8:
	max.s32 	%r119, %r11, 0;
	bra.uni 	LBB0_9;

LBB0_7:
	rem.s32 	%r74, %r11, %r58;
	add.s32 	%r75, %r74, %r58;
	rem.s32 	%r119, %r75, %r58;

LBB0_9:
	setp.lt.s32 	%p11, %r1, 1;
	mov.f32 	%f13, 0f00000000;
	and.pred  	%p13, %p11, %p6;
	mov.f32 	%f14, %f13;
	mov.f32 	%f15, %f13;
	@%p13 bra 	LBB0_11;

	add.s32 	%r76, %r119, %r5;
	mul.wide.s32 	%rd20, %r76, 4;
	add.s64 	%rd21, %rd3, %rd20;
	add.s64 	%rd22, %rd2, %rd20;
	add.s64 	%rd23, %rd1, %rd20;
	ld.global.nc.f32 	%f15, [%rd23];
	ld.global.nc.f32 	%f14, [%rd22];
	ld.global.nc.f32 	%f13, [%rd21];

LBB0_11:
	add.s32 	%r15, %r1, 1;
	@%p6 bra 	LBB0_13;
	bra.uni 	LBB0_12;

LBB0_13:
	add.s32 	%r79, %r58, -1;
	min.s32 	%r120, %r15, %r79;
	bra.uni 	LBB0_14;

LBB0_12:
	rem.s32 	%r77, %r15, %r58;
	add.s32 	%r78, %r77, %r58;
	rem.s32 	%r120, %r78, %r58;

LBB0_14:
	setp.ge.s32 	%p15, %r15, %r58;
	mov.f32 	%f19, 0f00000000;
	and.pred  	%p17, %p15, %p6;
	mov.f32 	%f20, %f19;
	mov.f32 	%f21, %f19;
	@%p17 bra 	LBB0_16;

	add.s32 	%r80, %r120, %r5;
	mul.wide.s32 	%rd24, %r80, 4;
	add.s64 	%rd25, %rd3, %rd24;
	add.s64 	%rd26, %rd2, %rd24;
	add.s64 	%rd27, %rd1, %rd24;
	ld.global.nc.f32 	%f21, [%rd27];
	ld.global.nc.f32 	%f20, [%rd26];
	ld.global.nc.f32 	%f19, [%rd25];

LBB0_16:
	add.s32 	%r19, %r1, 2;
	@%p6 bra 	LBB0_18;
	bra.uni 	LBB0_17;

LBB0_18:
	add.s32 	%r83, %r58, -1;
	min.s32 	%r121, %r19, %r83;
	bra.uni 	LBB0_19;

LBB0_17:
	rem.s32 	%r81, %r19, %r58;
	add.s32 	%r82, %r81, %r58;
	rem.s32 	%r121, %r82, %r58;

LBB0_19:
	add.s32 	%r23, %r121, %r5;
	setp.ge.s32 	%p19, %r19, %r58;
	mov.f32 	%f25, 0f00000000;
	and.pred  	%p21, %p19, %p6;
	mov.f32 	%f26, %f25;
	mov.f32 	%f27, %f25;
	@%p21 bra 	LBB0_21;

	mul.wide.s32 	%rd28, %r23, 4;
	add.s64 	%rd29, %rd3, %rd28;
	add.s64 	%rd30, %rd2, %rd28;
	add.s64 	%rd31, %rd1, %rd28;
	ld.global.nc.f32 	%f27, [%rd31];
	ld.global.nc.f32 	%f26, [%rd30];
	ld.global.nc.f32 	%f25, [%rd29];

LBB0_21:
	mul.f32 	%f208, %f20, %f20;
	fma.rn.f32 	%f209, %f19, %f19, %f208;
	fma.rn.f32 	%f31, %f21, %f21, %f209;
	setp.neu.f32 	%p22, %f31, 0f00000000;
	@%p22 bra 	LBB0_23;

	mul.f32 	%f213, %f14, %f14;
	fma.rn.f32 	%f214, %f13, %f13, %f213;
	fma.rn.f32 	%f215, %f15, %f15, %f214;
	setp.eq.f32 	%p23, %f215, 0f00000000;
	mov.f32 	%f427, 0f00000000;
	mov.f32 	%f428, %f427;
	mov.f32 	%f429, %f427;
	@%p23 bra 	LBB0_35;

LBB0_23:
	mul.f32 	%f216, %f8, %f8;
	fma.rn.f32 	%f217, %f7, %f7, %f216;
	fma.rn.f32 	%f38, %f9, %f9, %f217;
	setp.neu.f32 	%p24, %f38, 0f00000000;
	mul.f32 	%f218, %f26, %f26;
	fma.rn.f32 	%f219, %f25, %f25, %f218;
	fma.rn.f32 	%f42, %f27, %f27, %f219;
	setp.neu.f32 	%p25, %f42, 0f00000000;
	and.pred  	%p26, %p24, %p25;
	setp.eq.f32 	%p27, %f31, 0f00000000;
	or.pred  	%p28, %p27, %p26;
	@%p28 bra 	LBB0_25;

	mul.f32 	%f220, %f14, %f14;
	fma.rn.f32 	%f221, %f13, %f13, %f220;
	fma.rn.f32 	%f222, %f15, %f15, %f221;
	setp.neu.f32 	%p29, %f222, 0f00000000;
	@%p29 bra 	LBB0_34;
	bra.uni 	LBB0_25;

LBB0_34:
	sub.f32 	%f255, %f19, %f13;
	mul.f32 	%f427, %f255, 0f3F000000;
	sub.f32 	%f256, %f20, %f14;
	mul.f32 	%f428, %f256, 0f3F000000;
	sub.f32 	%f257, %f21, %f15;
	mul.f32 	%f429, %f257, 0f3F000000;
	bra.uni 	LBB0_35;

LBB0_25:
	setp.eq.f32 	%p30, %f38, 0f00000000;
	and.pred  	%p32, %p30, %p27;
	@%p32 bra 	LBB0_33;
	bra.uni 	LBB0_26;

LBB0_33:
	sub.f32 	%f427, %f1, %f13;
	sub.f32 	%f428, %f2, %f14;
	sub.f32 	%f429, %f3, %f15;
	bra.uni 	LBB0_35;

LBB0_26:
	setp.eq.f32 	%p33, %f42, 0f00000000;
	mul.f32 	%f223, %f14, %f14;
	fma.rn.f32 	%f224, %f13, %f13, %f223;
	fma.rn.f32 	%f43, %f15, %f15, %f224;
	setp.eq.f32 	%p34, %f43, 0f00000000;
	and.pred  	%p35, %p34, %p33;
	@%p35 bra 	LBB0_32;
	bra.uni 	LBB0_27;

LBB0_32:
	sub.f32 	%f427, %f19, %f1;
	sub.f32 	%f428, %f20, %f2;
	sub.f32 	%f429, %f21, %f3;
	bra.uni 	LBB0_35;

LBB0_27:
	or.pred  	%p38, %p30, %p22;
	@%p38 bra 	LBB0_29;
	bra.uni 	LBB0_28;

LBB0_29:
	setp.neu.f32 	%p39, %f43, 0f00000000;
	or.pred  	%p41, %p33, %p39;
	@%p41 bra 	LBB0_31;
	bra.uni 	LBB0_30;

LBB0_31:
	sub.f32 	%f246, %f19, %f13;
	sub.f32 	%f247, %f20, %f14;
	sub.f32 	%f248, %f21, %f15;
	sub.f32 	%f249, %f7, %f25;
	mul.f32 	%f250, %f249, 0f3DAAAAAB;
	sub.f32 	%f251, %f8, %f26;
	mul.f32 	%f252, %f251, 0f3DAAAAAB;
	sub.f32 	%f253, %f9, %f27;
	mul.f32 	%f254, %f253, 0f3DAAAAAB;
	fma.rn.f32 	%f427, %f246, 0f3F2AAAAB, %f250;
	fma.rn.f32 	%f428, %f247, 0f3F2AAAAB, %f252;
	fma.rn.f32 	%f429, %f248, 0f3F2AAAAB, %f254;
	bra.uni 	LBB0_35;

LBB0_28:
	mul.f32 	%f225, %f7, 0f3F000000;
	add.f32 	%f226, %f13, %f13;
	sub.f32 	%f227, %f225, %f226;
	add.f32 	%f228, %f14, %f14;
	mul.f32 	%f229, %f8, 0f3F000000;
	sub.f32 	%f230, %f229, %f228;
	add.f32 	%f231, %f15, %f15;
	mul.f32 	%f232, %f9, 0f3F000000;
	sub.f32 	%f233, %f232, %f231;
	fma.rn.f32 	%f427, %f1, 0f3FC00000, %f227;
	fma.rn.f32 	%f428, %f2, 0f3FC00000, %f230;
	fma.rn.f32 	%f429, %f3, 0f3FC00000, %f233;
	bra.uni 	LBB0_35;

LBB0_30:
	mul.f32 	%f234, %f25, 0f3F000000;
	add.f32 	%f235, %f19, %f19;
	sub.f32 	%f236, %f235, %f234;
	add.f32 	%f237, %f20, %f20;
	mul.f32 	%f238, %f26, 0f3F000000;
	sub.f32 	%f239, %f237, %f238;
	add.f32 	%f240, %f21, %f21;
	mul.f32 	%f241, %f27, 0f3F000000;
	sub.f32 	%f242, %f240, %f241;
	mul.f32 	%f243, %f1, 0f3FC00000;
	sub.f32 	%f427, %f236, %f243;
	mul.f32 	%f244, %f2, 0f3FC00000;
	sub.f32 	%f428, %f239, %f244;
	mul.f32 	%f245, %f3, 0f3FC00000;
	sub.f32 	%f429, %f242, %f245;

LBB0_35:
	and.b16  	%rs2, %rs4, 2;
	setp.eq.s16 	%p42, %rs2, 0;
	add.s32 	%r24, %r2, -2;
	@%p42 bra 	LBB0_37;
	bra.uni 	LBB0_36;

LBB0_37:
	max.s32 	%r122, %r24, 0;
	bra.uni 	LBB0_38;

LBB0_36:
	rem.s32 	%r84, %r24, %r59;
	add.s32 	%r85, %r84, %r59;
	rem.s32 	%r122, %r85, %r59;

LBB0_38:
	setp.lt.s32 	%p44, %r2, 2;
	mov.f32 	%f68, 0f00000000;
	and.pred  	%p45, %p44, %p42;
	mov.f32 	%f69, %f68;
	mov.f32 	%f70, %f68;
	@%p45 bra 	LBB0_40;

	add.s32 	%r86, %r122, %r4;
	mad.lo.s32 	%r87, %r86, %r58, %r1;
	mul.wide.s32 	%rd32, %r87, 4;
	add.s64 	%rd33, %rd3, %rd32;
	add.s64 	%rd34, %rd2, %rd32;
	add.s64 	%rd35, %rd1, %rd32;
	ld.global.nc.f32 	%f70, [%rd35];
	ld.global.nc.f32 	%f69, [%rd34];
	ld.global.nc.f32 	%f68, [%rd33];

LBB0_40:
	add.s32 	%r28, %r2, -1;
	@%p42 bra 	LBB0_42;
	bra.uni 	LBB0_41;

LBB0_42:
	max.s32 	%r123, %r28, 0;
	bra.uni 	LBB0_43;

LBB0_41:
	rem.s32 	%r88, %r28, %r59;
	add.s32 	%r89, %r88, %r59;
	rem.s32 	%r123, %r89, %r59;

LBB0_43:
	setp.lt.s32 	%p47, %r2, 1;
	mov.f32 	%f74, 0f00000000;
	and.pred  	%p49, %p47, %p42;
	mov.f32 	%f75, %f74;
	mov.f32 	%f76, %f74;
	@%p49 bra 	LBB0_45;

	add.s32 	%r90, %r123, %r4;
	mad.lo.s32 	%r91, %r90, %r58, %r1;
	mul.wide.s32 	%rd36, %r91, 4;
	add.s64 	%rd37, %rd3, %rd36;
	add.s64 	%rd38, %rd2, %rd36;
	add.s64 	%rd39, %rd1, %rd36;
	ld.global.nc.f32 	%f76, [%rd39];
	ld.global.nc.f32 	%f75, [%rd38];
	ld.global.nc.f32 	%f74, [%rd37];

LBB0_45:
	add.s32 	%r32, %r2, 1;
	@%p42 bra 	LBB0_47;
	bra.uni 	LBB0_46;

LBB0_47:
	add.s32 	%r94, %r59, -1;
	min.s32 	%r124, %r32, %r94;
	bra.uni 	LBB0_48;

LBB0_46:
	rem.s32 	%r92, %r32, %r59;
	add.s32 	%r93, %r92, %r59;
	rem.s32 	%r124, %r93, %r59;

LBB0_48:
	setp.ge.s32 	%p51, %r32, %r59;
	mov.f32 	%f80, 0f00000000;
	and.pred  	%p53, %p51, %p42;
	mov.f32 	%f81, %f80;
	mov.f32 	%f82, %f80;
	@%p53 bra 	LBB0_50;

	add.s32 	%r95, %r124, %r4;
	mad.lo.s32 	%r96, %r95, %r58, %r1;
	mul.wide.s32 	%rd40, %r96, 4;
	add.s64 	%rd41, %rd3, %rd40;
	add.s64 	%rd42, %rd2, %rd40;
	add.s64 	%rd43, %rd1, %rd40;
	ld.global.nc.f32 	%f82, [%rd43];
	ld.global.nc.f32 	%f81, [%rd42];
	ld.global.nc.f32 	%f80, [%rd41];

LBB0_50:
	add.s32 	%r36, %r2, 2;
	@%p42 bra 	LBB0_52;
	bra.uni 	LBB0_51;

LBB0_52:
	add.s32 	%r99, %r59, -1;
	min.s32 	%r125, %r36, %r99;
	bra.uni 	LBB0_53;

LBB0_51:
	rem.s32 	%r97, %r36, %r59;
	add.s32 	%r98, %r97, %r59;
	rem.s32 	%r125, %r98, %r59;

LBB0_53:
	add.s32 	%r40, %r125, %r4;
	setp.ge.s32 	%p55, %r36, %r59;
	mov.f32 	%f86, 0f00000000;
	and.pred  	%p57, %p55, %p42;
	mov.f32 	%f87, %f86;
	mov.f32 	%f88, %f86;
	@%p57 bra 	LBB0_55;

	mad.lo.s32 	%r100, %r40, %r58, %r1;
	mul.wide.s32 	%rd44, %r100, 4;
	add.s64 	%rd45, %rd3, %rd44;
	add.s64 	%rd46, %rd2, %rd44;
	add.s64 	%rd47, %rd1, %rd44;
	ld.global.nc.f32 	%f88, [%rd47];
	ld.global.nc.f32 	%f87, [%rd46];
	ld.global.nc.f32 	%f86, [%rd45];

LBB0_55:
	mul.f32 	%f270, %f81, %f81;
	fma.rn.f32 	%f271, %f80, %f80, %f270;
	fma.rn.f32 	%f92, %f82, %f82, %f271;
	setp.neu.f32 	%p58, %f92, 0f00000000;
	@%p58 bra 	LBB0_57;

	mul.f32 	%f275, %f75, %f75;
	fma.rn.f32 	%f276, %f74, %f74, %f275;
	fma.rn.f32 	%f277, %f76, %f76, %f276;
	setp.eq.f32 	%p59, %f277, 0f00000000;
	mov.f32 	%f442, 0f00000000;
	mov.f32 	%f443, %f442;
	mov.f32 	%f444, %f442;
	@%p59 bra 	LBB0_69;

LBB0_57:
	mul.f32 	%f278, %f69, %f69;
	fma.rn.f32 	%f279, %f68, %f68, %f278;
	fma.rn.f32 	%f99, %f70, %f70, %f279;
	setp.neu.f32 	%p60, %f99, 0f00000000;
	mul.f32 	%f280, %f87, %f87;
	fma.rn.f32 	%f281, %f86, %f86, %f280;
	fma.rn.f32 	%f103, %f88, %f88, %f281;
	setp.neu.f32 	%p61, %f103, 0f00000000;
	and.pred  	%p62, %p60, %p61;
	setp.eq.f32 	%p63, %f92, 0f00000000;
	or.pred  	%p64, %p63, %p62;
	@%p64 bra 	LBB0_59;

	mul.f32 	%f282, %f75, %f75;
	fma.rn.f32 	%f283, %f74, %f74, %f282;
	fma.rn.f32 	%f284, %f76, %f76, %f283;
	setp.neu.f32 	%p65, %f284, 0f00000000;
	@%p65 bra 	LBB0_68;
	bra.uni 	LBB0_59;

LBB0_68:
	sub.f32 	%f317, %f80, %f74;
	mul.f32 	%f442, %f317, 0f3F000000;
	sub.f32 	%f318, %f81, %f75;
	mul.f32 	%f443, %f318, 0f3F000000;
	sub.f32 	%f319, %f82, %f76;
	mul.f32 	%f444, %f319, 0f3F000000;
	bra.uni 	LBB0_69;

LBB0_59:
	setp.eq.f32 	%p66, %f99, 0f00000000;
	and.pred  	%p68, %p66, %p63;
	@%p68 bra 	LBB0_67;
	bra.uni 	LBB0_60;

LBB0_67:
	sub.f32 	%f442, %f1, %f74;
	sub.f32 	%f443, %f2, %f75;
	sub.f32 	%f444, %f3, %f76;
	bra.uni 	LBB0_69;

LBB0_60:
	setp.eq.f32 	%p69, %f103, 0f00000000;
	mul.f32 	%f285, %f75, %f75;
	fma.rn.f32 	%f286, %f74, %f74, %f285;
	fma.rn.f32 	%f104, %f76, %f76, %f286;
	setp.eq.f32 	%p70, %f104, 0f00000000;
	and.pred  	%p71, %p70, %p69;
	@%p71 bra 	LBB0_66;
	bra.uni 	LBB0_61;

LBB0_66:
	sub.f32 	%f442, %f80, %f1;
	sub.f32 	%f443, %f81, %f2;
	sub.f32 	%f444, %f82, %f3;
	bra.uni 	LBB0_69;

LBB0_61:
	or.pred  	%p74, %p66, %p58;
	@%p74 bra 	LBB0_63;
	bra.uni 	LBB0_62;

LBB0_63:
	setp.neu.f32 	%p75, %f104, 0f00000000;
	or.pred  	%p77, %p69, %p75;
	@%p77 bra 	LBB0_65;
	bra.uni 	LBB0_64;

LBB0_65:
	sub.f32 	%f308, %f80, %f74;
	sub.f32 	%f309, %f81, %f75;
	sub.f32 	%f310, %f82, %f76;
	sub.f32 	%f311, %f68, %f86;
	mul.f32 	%f312, %f311, 0f3DAAAAAB;
	sub.f32 	%f313, %f69, %f87;
	mul.f32 	%f314, %f313, 0f3DAAAAAB;
	sub.f32 	%f315, %f70, %f88;
	mul.f32 	%f316, %f315, 0f3DAAAAAB;
	fma.rn.f32 	%f442, %f308, 0f3F2AAAAB, %f312;
	fma.rn.f32 	%f443, %f309, 0f3F2AAAAB, %f314;
	fma.rn.f32 	%f444, %f310, 0f3F2AAAAB, %f316;
	bra.uni 	LBB0_69;

LBB0_62:
	mul.f32 	%f287, %f68, 0f3F000000;
	add.f32 	%f288, %f74, %f74;
	sub.f32 	%f289, %f287, %f288;
	add.f32 	%f290, %f75, %f75;
	mul.f32 	%f291, %f69, 0f3F000000;
	sub.f32 	%f292, %f291, %f290;
	add.f32 	%f293, %f76, %f76;
	mul.f32 	%f294, %f70, 0f3F000000;
	sub.f32 	%f295, %f294, %f293;
	fma.rn.f32 	%f442, %f1, 0f3FC00000, %f289;
	fma.rn.f32 	%f443, %f2, 0f3FC00000, %f292;
	fma.rn.f32 	%f444, %f3, 0f3FC00000, %f295;
	bra.uni 	LBB0_69;

LBB0_64:
	mul.f32 	%f296, %f86, 0f3F000000;
	add.f32 	%f297, %f80, %f80;
	sub.f32 	%f298, %f297, %f296;
	add.f32 	%f299, %f81, %f81;
	mul.f32 	%f300, %f87, 0f3F000000;
	sub.f32 	%f301, %f299, %f300;
	add.f32 	%f302, %f82, %f82;
	mul.f32 	%f303, %f88, 0f3F000000;
	sub.f32 	%f304, %f302, %f303;
	mul.f32 	%f305, %f1, 0f3FC00000;
	sub.f32 	%f442, %f298, %f305;
	mul.f32 	%f306, %f2, 0f3FC00000;
	sub.f32 	%f443, %f301, %f306;
	mul.f32 	%f307, %f3, 0f3FC00000;
	sub.f32 	%f444, %f304, %f307;

LBB0_69:
	and.b16  	%rs3, %rs4, 4;
	setp.eq.s16 	%p78, %rs3, 0;
	add.s32 	%r41, %r3, -2;
	@%p78 bra 	LBB0_71;
	bra.uni 	LBB0_70;

LBB0_71:
	max.s32 	%r126, %r41, 0;
	bra.uni 	LBB0_72;

LBB0_70:
	rem.s32 	%r101, %r41, %r60;
	add.s32 	%r102, %r101, %r60;
	rem.s32 	%r126, %r102, %r60;

LBB0_72:
	setp.lt.s32 	%p80, %r3, 2;
	mov.f32 	%f129, 0f00000000;
	and.pred  	%p81, %p80, %p78;
	mov.f32 	%f130, %f129;
	mov.f32 	%f131, %f129;
	@%p81 bra 	LBB0_74;

	mad.lo.s32 	%r103, %r126, %r59, %r2;
	mad.lo.s32 	%r104, %r103, %r58, %r1;
	mul.wide.s32 	%rd48, %r104, 4;
	add.s64 	%rd49, %rd3, %rd48;
	add.s64 	%rd50, %rd2, %rd48;
	add.s64 	%rd51, %rd1, %rd48;
	ld.global.nc.f32 	%f131, [%rd51];
	ld.global.nc.f32 	%f130, [%rd50];
	ld.global.nc.f32 	%f129, [%rd49];

LBB0_74:
	add.s32 	%r45, %r3, -1;
	@%p78 bra 	LBB0_76;
	bra.uni 	LBB0_75;

LBB0_76:
	max.s32 	%r127, %r45, 0;
	bra.uni 	LBB0_77;

LBB0_75:
	rem.s32 	%r105, %r45, %r60;
	add.s32 	%r106, %r105, %r60;
	rem.s32 	%r127, %r106, %r60;

LBB0_77:
	setp.lt.s32 	%p83, %r3, 1;
	mov.f32 	%f135, 0f00000000;
	and.pred  	%p85, %p83, %p78;
	mov.f32 	%f136, %f135;
	mov.f32 	%f137, %f135;
	@%p85 bra 	LBB0_79;

	mad.lo.s32 	%r107, %r127, %r59, %r2;
	mad.lo.s32 	%r108, %r107, %r58, %r1;
	mul.wide.s32 	%rd52, %r108, 4;
	add.s64 	%rd53, %rd3, %rd52;
	add.s64 	%rd54, %rd2, %rd52;
	add.s64 	%rd55, %rd1, %rd52;
	ld.global.nc.f32 	%f137, [%rd55];
	ld.global.nc.f32 	%f136, [%rd54];
	ld.global.nc.f32 	%f135, [%rd53];

LBB0_79:
	add.s32 	%r49, %r3, 1;
	@%p78 bra 	LBB0_81;
	bra.uni 	LBB0_80;

LBB0_81:
	add.s32 	%r111, %r60, -1;
	min.s32 	%r128, %r49, %r111;
	bra.uni 	LBB0_82;

LBB0_80:
	rem.s32 	%r109, %r49, %r60;
	add.s32 	%r110, %r109, %r60;
	rem.s32 	%r128, %r110, %r60;

LBB0_82:
	setp.ge.s32 	%p87, %r49, %r60;
	mov.f32 	%f141, 0f00000000;
	and.pred  	%p89, %p87, %p78;
	mov.f32 	%f142, %f141;
	mov.f32 	%f143, %f141;
	@%p89 bra 	LBB0_84;

	mad.lo.s32 	%r112, %r128, %r59, %r2;
	mad.lo.s32 	%r113, %r112, %r58, %r1;
	mul.wide.s32 	%rd56, %r113, 4;
	add.s64 	%rd57, %rd3, %rd56;
	add.s64 	%rd58, %rd2, %rd56;
	add.s64 	%rd59, %rd1, %rd56;
	ld.global.nc.f32 	%f143, [%rd59];
	ld.global.nc.f32 	%f142, [%rd58];
	ld.global.nc.f32 	%f141, [%rd57];

LBB0_84:
	add.s32 	%r53, %r3, 2;
	@%p78 bra 	LBB0_86;
	bra.uni 	LBB0_85;

LBB0_86:
	add.s32 	%r116, %r60, -1;
	min.s32 	%r129, %r53, %r116;
	bra.uni 	LBB0_87;

LBB0_85:
	rem.s32 	%r114, %r53, %r60;
	add.s32 	%r115, %r114, %r60;
	rem.s32 	%r129, %r115, %r60;

LBB0_87:
	mad.lo.s32 	%r117, %r129, %r59, %r2;
	mad.lo.s32 	%r57, %r117, %r58, %r1;
	setp.ge.s32 	%p91, %r53, %r60;
	mov.f32 	%f147, 0f00000000;
	and.pred  	%p93, %p91, %p78;
	mov.f32 	%f148, %f147;
	mov.f32 	%f149, %f147;
	@%p93 bra 	LBB0_89;

	mul.wide.s32 	%rd60, %r57, 4;
	add.s64 	%rd61, %rd3, %rd60;
	add.s64 	%rd62, %rd2, %rd60;
	add.s64 	%rd63, %rd1, %rd60;
	ld.global.nc.f32 	%f149, [%rd63];
	ld.global.nc.f32 	%f148, [%rd62];
	ld.global.nc.f32 	%f147, [%rd61];

LBB0_89:
	mul.f32 	%f332, %f142, %f142;
	fma.rn.f32 	%f333, %f141, %f141, %f332;
	fma.rn.f32 	%f153, %f143, %f143, %f333;
	setp.neu.f32 	%p94, %f153, 0f00000000;
	@%p94 bra 	LBB0_91;

	mul.f32 	%f337, %f136, %f136;
	fma.rn.f32 	%f338, %f135, %f135, %f337;
	fma.rn.f32 	%f339, %f137, %f137, %f338;
	setp.eq.f32 	%p95, %f339, 0f00000000;
	mov.f32 	%f457, 0f00000000;
	mov.f32 	%f458, %f457;
	mov.f32 	%f459, %f457;
	@%p95 bra 	LBB0_103;

LBB0_91:
	mul.f32 	%f340, %f130, %f130;
	fma.rn.f32 	%f341, %f129, %f129, %f340;
	fma.rn.f32 	%f160, %f131, %f131, %f341;
	setp.neu.f32 	%p96, %f160, 0f00000000;
	mul.f32 	%f342, %f148, %f148;
	fma.rn.f32 	%f343, %f147, %f147, %f342;
	fma.rn.f32 	%f164, %f149, %f149, %f343;
	setp.neu.f32 	%p97, %f164, 0f00000000;
	and.pred  	%p98, %p96, %p97;
	setp.eq.f32 	%p99, %f153, 0f00000000;
	or.pred  	%p100, %p99, %p98;
	@%p100 bra 	LBB0_93;

	mul.f32 	%f344, %f136, %f136;
	fma.rn.f32 	%f345, %f135, %f135, %f344;
	fma.rn.f32 	%f346, %f137, %f137, %f345;
	setp.neu.f32 	%p101, %f346, 0f00000000;
	@%p101 bra 	LBB0_102;
	bra.uni 	LBB0_93;

LBB0_102:
	sub.f32 	%f379, %f141, %f135;
	mul.f32 	%f457, %f379, 0f3F000000;
	sub.f32 	%f380, %f142, %f136;
	mul.f32 	%f458, %f380, 0f3F000000;
	sub.f32 	%f381, %f143, %f137;
	mul.f32 	%f459, %f381, 0f3F000000;
	bra.uni 	LBB0_103;

LBB0_93:
	setp.eq.f32 	%p102, %f160, 0f00000000;
	and.pred  	%p104, %p102, %p99;
	@%p104 bra 	LBB0_101;
	bra.uni 	LBB0_94;

LBB0_101:
	sub.f32 	%f457, %f1, %f135;
	sub.f32 	%f458, %f2, %f136;
	sub.f32 	%f459, %f3, %f137;
	bra.uni 	LBB0_103;

LBB0_94:
	setp.eq.f32 	%p105, %f164, 0f00000000;
	mul.f32 	%f347, %f136, %f136;
	fma.rn.f32 	%f348, %f135, %f135, %f347;
	fma.rn.f32 	%f165, %f137, %f137, %f348;
	setp.eq.f32 	%p106, %f165, 0f00000000;
	and.pred  	%p107, %p106, %p105;
	@%p107 bra 	LBB0_100;
	bra.uni 	LBB0_95;

LBB0_100:
	sub.f32 	%f457, %f141, %f1;
	sub.f32 	%f458, %f142, %f2;
	sub.f32 	%f459, %f143, %f3;
	bra.uni 	LBB0_103;

LBB0_95:
	or.pred  	%p110, %p102, %p94;
	@%p110 bra 	LBB0_97;
	bra.uni 	LBB0_96;

LBB0_97:
	setp.neu.f32 	%p111, %f165, 0f00000000;
	or.pred  	%p113, %p105, %p111;
	@%p113 bra 	LBB0_99;
	bra.uni 	LBB0_98;

LBB0_99:
	sub.f32 	%f370, %f141, %f135;
	sub.f32 	%f371, %f142, %f136;
	sub.f32 	%f372, %f143, %f137;
	sub.f32 	%f373, %f129, %f147;
	mul.f32 	%f374, %f373, 0f3DAAAAAB;
	sub.f32 	%f375, %f130, %f148;
	mul.f32 	%f376, %f375, 0f3DAAAAAB;
	sub.f32 	%f377, %f131, %f149;
	mul.f32 	%f378, %f377, 0f3DAAAAAB;
	fma.rn.f32 	%f457, %f370, 0f3F2AAAAB, %f374;
	fma.rn.f32 	%f458, %f371, 0f3F2AAAAB, %f376;
	fma.rn.f32 	%f459, %f372, 0f3F2AAAAB, %f378;
	bra.uni 	LBB0_103;

LBB0_96:
	mul.f32 	%f349, %f129, 0f3F000000;
	add.f32 	%f350, %f135, %f135;
	sub.f32 	%f351, %f349, %f350;
	add.f32 	%f352, %f136, %f136;
	mul.f32 	%f353, %f130, 0f3F000000;
	sub.f32 	%f354, %f353, %f352;
	add.f32 	%f355, %f137, %f137;
	mul.f32 	%f356, %f131, 0f3F000000;
	sub.f32 	%f357, %f356, %f355;
	fma.rn.f32 	%f457, %f1, 0f3FC00000, %f351;
	fma.rn.f32 	%f458, %f2, 0f3FC00000, %f354;
	fma.rn.f32 	%f459, %f3, 0f3FC00000, %f357;
	bra.uni 	LBB0_103;

LBB0_98:
	mul.f32 	%f358, %f147, 0f3F000000;
	add.f32 	%f359, %f141, %f141;
	sub.f32 	%f360, %f359, %f358;
	add.f32 	%f361, %f142, %f142;
	mul.f32 	%f362, %f148, 0f3F000000;
	sub.f32 	%f363, %f361, %f362;
	add.f32 	%f364, %f143, %f143;
	mul.f32 	%f365, %f149, 0f3F000000;
	sub.f32 	%f366, %f364, %f365;
	mul.f32 	%f367, %f1, 0f3FC00000;
	sub.f32 	%f457, %f360, %f367;
	mul.f32 	%f368, %f2, 0f3FC00000;
	sub.f32 	%f458, %f363, %f368;
	mul.f32 	%f369, %f3, 0f3FC00000;
	sub.f32 	%f459, %f366, %f369;

LBB0_103:
	setp.eq.s64 	%p114, %rd7, 0;
	@%p114 bra 	LBB0_105;

	cvta.to.global.u64 	%rd64, %rd7;
	add.s64 	%rd66, %rd64, %rd12;
	ld.global.nc.f32 	%f382, [%rd66];
	mul.f32 	%f460, %f382, %f460;

LBB0_105:
	setp.eq.s64 	%p115, %rd8, 0;
	@%p115 bra 	LBB0_107;

	cvta.to.global.u64 	%rd67, %rd8;
	add.s64 	%rd69, %rd67, %rd12;
	ld.global.nc.f32 	%f383, [%rd69];
	mul.f32 	%f461, %f383, %f461;

LBB0_107:
	mul.f32 	%f384, %f427, %f193;
	mul.f32 	%f385, %f443, %f194;
	mul.f32 	%f386, %f459, %f195;
	add.f32 	%f387, %f460, %f460;
	mul.f32 	%f388, %f1, %f387;
	add.f32 	%f389, %f385, %f386;
	mul.f32 	%f390, %f1, %f389;
	mul.f32 	%f391, %f442, %f194;
	fma.rn.f32 	%f392, %f2, %f391, %f390;
	mul.f32 	%f393, %f457, %f195;
	fma.rn.f32 	%f394, %f3, %f393, %f392;
	mul.f32 	%f395, %f394, %f461;
	fma.rn.f32 	%f396, %f384, %f388, %f395;
	cvta.to.global.u64 	%rd70, %rd4;
	add.s64 	%rd72, %rd70, %rd12;
	st.global.f32 	[%rd72], %f396;
	mul.f32 	%f397, %f2, %f387;
	add.f32 	%f398, %f384, %f386;
	mul.f32 	%f399, %f2, %f398;
	mul.f32 	%f400, %f428, %f193;
	fma.rn.f32 	%f401, %f1, %f400, %f399;
	mul.f32 	%f402, %f458, %f195;
	fma.rn.f32 	%f403, %f3, %f402, %f401;
	mul.f32 	%f404, %f403, %f461;
	fma.rn.f32 	%f405, %f385, %f397, %f404;
	cvta.to.global.u64 	%rd73, %rd5;
	add.s64 	%rd74, %rd73, %rd12;
	st.global.f32 	[%rd74], %f405;
	mul.f32 	%f406, %f3, %f387;
	mul.f32 	%f407, %f444, %f194;
	mul.f32 	%f408, %f2, %f407;
	mul.f32 	%f409, %f429, %f193;
	fma.rn.f32 	%f410, %f1, %f409, %f408;
	add.f32 	%f411, %f384, %f385;
	fma.rn.f32 	%f412, %f3, %f411, %f410;
	mul.f32 	%f413, %f412, %f461;
	fma.rn.f32 	%f414, %f386, %f406, %f413;
	cvta.to.global.u64 	%rd75, %rd6;
	add.s64 	%rd76, %rd75, %rd12;
	st.global.f32 	[%rd76], %f414;

LBB0_108:
	ret;

}

`
)
