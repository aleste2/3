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

// CUDA handle for reducemaxvecdiff2 kernel
var reducemaxvecdiff2_code cu.Function

// Stores the arguments for reducemaxvecdiff2 kernel invocation
type reducemaxvecdiff2_args_t struct {
	arg_x1      unsafe.Pointer
	arg_y1      unsafe.Pointer
	arg_z1      unsafe.Pointer
	arg_x2      unsafe.Pointer
	arg_y2      unsafe.Pointer
	arg_z2      unsafe.Pointer
	arg_dst     unsafe.Pointer
	arg_initVal float32
	arg_n       int
	argptr      [9]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for reducemaxvecdiff2 kernel invocation
var reducemaxvecdiff2_args reducemaxvecdiff2_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	reducemaxvecdiff2_args.argptr[0] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_x1)
	reducemaxvecdiff2_args.argptr[1] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_y1)
	reducemaxvecdiff2_args.argptr[2] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_z1)
	reducemaxvecdiff2_args.argptr[3] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_x2)
	reducemaxvecdiff2_args.argptr[4] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_y2)
	reducemaxvecdiff2_args.argptr[5] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_z2)
	reducemaxvecdiff2_args.argptr[6] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_dst)
	reducemaxvecdiff2_args.argptr[7] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_initVal)
	reducemaxvecdiff2_args.argptr[8] = unsafe.Pointer(&reducemaxvecdiff2_args.arg_n)
}

// Wrapper for reducemaxvecdiff2 CUDA kernel, asynchronous.
func k_reducemaxvecdiff2_async(x1 unsafe.Pointer, y1 unsafe.Pointer, z1 unsafe.Pointer, x2 unsafe.Pointer, y2 unsafe.Pointer, z2 unsafe.Pointer, dst unsafe.Pointer, initVal float32, n int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("reducemaxvecdiff2")
	}

	reducemaxvecdiff2_args.Lock()
	defer reducemaxvecdiff2_args.Unlock()

	if reducemaxvecdiff2_code == 0 {
		reducemaxvecdiff2_code = fatbinLoad(reducemaxvecdiff2_map, "reducemaxvecdiff2")
	}

	reducemaxvecdiff2_args.arg_x1 = x1
	reducemaxvecdiff2_args.arg_y1 = y1
	reducemaxvecdiff2_args.arg_z1 = z1
	reducemaxvecdiff2_args.arg_x2 = x2
	reducemaxvecdiff2_args.arg_y2 = y2
	reducemaxvecdiff2_args.arg_z2 = z2
	reducemaxvecdiff2_args.arg_dst = dst
	reducemaxvecdiff2_args.arg_initVal = initVal
	reducemaxvecdiff2_args.arg_n = n

	args := reducemaxvecdiff2_args.argptr[:]
	cu.LaunchKernel(reducemaxvecdiff2_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("reducemaxvecdiff2")
	}
}

// maps compute capability on PTX code for reducemaxvecdiff2 kernel.
var reducemaxvecdiff2_map = map[int]string{0: "",
	70: reducemaxvecdiff2_ptx_70}

// reducemaxvecdiff2 PTX code for various compute capabilities.
const (
	reducemaxvecdiff2_ptx_70 = `
.version 7.2
.target sm_70
.address_size 64

	// .globl	reducemaxvecdiff2

.visible .entry reducemaxvecdiff2(
	.param .u64 reducemaxvecdiff2_param_0,
	.param .u64 reducemaxvecdiff2_param_1,
	.param .u64 reducemaxvecdiff2_param_2,
	.param .u64 reducemaxvecdiff2_param_3,
	.param .u64 reducemaxvecdiff2_param_4,
	.param .u64 reducemaxvecdiff2_param_5,
	.param .u64 reducemaxvecdiff2_param_6,
	.param .f32 reducemaxvecdiff2_param_7,
	.param .u32 reducemaxvecdiff2_param_8
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<101>;
	.reg .b32 	%r<69>;
	.reg .b64 	%rd<73>;
	// demoted variable
	.shared .align 4 .b8 _ZZ17reducemaxvecdiff2E5sdata[2048];

	ld.param.u64 	%rd20, [reducemaxvecdiff2_param_0];
	ld.param.u64 	%rd21, [reducemaxvecdiff2_param_1];
	ld.param.u64 	%rd22, [reducemaxvecdiff2_param_2];
	ld.param.u64 	%rd23, [reducemaxvecdiff2_param_3];
	ld.param.u64 	%rd24, [reducemaxvecdiff2_param_4];
	ld.param.u64 	%rd25, [reducemaxvecdiff2_param_5];
	ld.param.u64 	%rd26, [reducemaxvecdiff2_param_6];
	ld.param.f32 	%f100, [reducemaxvecdiff2_param_7];
	ld.param.u32 	%r15, [reducemaxvecdiff2_param_8];
	mov.u32 	%r16, %tid.x;
	mov.u32 	%r68, %ntid.x;
	mov.u32 	%r18, %ctaid.x;
	mad.lo.s32 	%r19, %r18, %r68, %r16;
	setp.ge.s32 	%p1, %r19, %r15;
	@%p1 bra 	LBB0_7;

	mov.u32 	%r20, %nctaid.x;
	mov.u32 	%r21, %ntid.x;
	mul.lo.s32 	%r22, %r20, %r21;
	add.s32 	%r23, %r22, %r15;
	mad.lo.s32 	%r66, %r18, %r21, %r16;
	add.s32 	%r26, %r66, %r22;
	not.b32 	%r27, %r26;
	add.s32 	%r28, %r23, %r27;
	div.u32 	%r2, %r28, %r22;
	add.s32 	%r29, %r2, 1;
	and.b32  	%r65, %r29, 3;
	setp.eq.s32 	%p2, %r65, 0;
	@%p2 bra 	LBB0_4;

	mad.lo.s32 	%r66, %r18, %r21, %r16;
	cvta.to.global.u64 	%rd27, %rd25;
	mul.wide.s32 	%rd28, %r66, 4;
	add.s64 	%rd72, %rd27, %rd28;
	cvta.to.global.u64 	%rd29, %rd22;
	add.s64 	%rd71, %rd29, %rd28;
	cvta.to.global.u64 	%rd30, %rd24;
	add.s64 	%rd70, %rd30, %rd28;
	cvta.to.global.u64 	%rd31, %rd21;
	add.s64 	%rd69, %rd31, %rd28;
	cvta.to.global.u64 	%rd32, %rd23;
	add.s64 	%rd68, %rd32, %rd28;
	cvta.to.global.u64 	%rd33, %rd20;
	add.s64 	%rd67, %rd33, %rd28;

LBB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd68];
	ld.global.nc.f32 	%f11, [%rd67];
	sub.f32 	%f12, %f11, %f10;
	ld.global.nc.f32 	%f13, [%rd70];
	ld.global.nc.f32 	%f14, [%rd69];
	sub.f32 	%f15, %f14, %f13;
	mul.f32 	%f16, %f15, %f15;
	fma.rn.f32 	%f17, %f12, %f12, %f16;
	ld.global.nc.f32 	%f18, [%rd72];
	ld.global.nc.f32 	%f19, [%rd71];
	sub.f32 	%f20, %f19, %f18;
	fma.rn.f32 	%f21, %f20, %f20, %f17;
	max.f32 	%f100, %f100, %f21;
	add.s32 	%r66, %r66, %r22;
	mul.wide.s32 	%rd34, %r22, 4;
	add.s64 	%rd72, %rd72, %rd34;
	add.s64 	%rd71, %rd71, %rd34;
	add.s64 	%rd70, %rd70, %rd34;
	add.s64 	%rd69, %rd69, %rd34;
	add.s64 	%rd68, %rd68, %rd34;
	add.s64 	%rd67, %rd67, %rd34;
	add.s32 	%r65, %r65, -1;
	setp.ne.s32 	%p3, %r65, 0;
	@%p3 bra 	LBB0_3;

LBB0_4:
	setp.lt.u32 	%p4, %r2, 3;
	@%p4 bra 	LBB0_7;

	cvta.to.global.u64 	%rd19, %rd20;
	cvta.to.global.u64 	%rd37, %rd23;
	cvta.to.global.u64 	%rd39, %rd21;

LBB0_6:
	mul.wide.s32 	%rd35, %r66, 4;
	add.s64 	%rd36, %rd19, %rd35;
	add.s64 	%rd38, %rd37, %rd35;
	ld.global.nc.f32 	%f22, [%rd38];
	ld.global.nc.f32 	%f23, [%rd36];
	sub.f32 	%f24, %f23, %f22;
	add.s64 	%rd40, %rd39, %rd35;
	cvta.to.global.u64 	%rd41, %rd24;
	add.s64 	%rd42, %rd41, %rd35;
	ld.global.nc.f32 	%f25, [%rd42];
	ld.global.nc.f32 	%f26, [%rd40];
	sub.f32 	%f27, %f26, %f25;
	mul.f32 	%f28, %f27, %f27;
	fma.rn.f32 	%f29, %f24, %f24, %f28;
	cvta.to.global.u64 	%rd43, %rd22;
	add.s64 	%rd44, %rd43, %rd35;
	cvta.to.global.u64 	%rd45, %rd25;
	add.s64 	%rd46, %rd45, %rd35;
	ld.global.nc.f32 	%f30, [%rd46];
	ld.global.nc.f32 	%f31, [%rd44];
	sub.f32 	%f32, %f31, %f30;
	fma.rn.f32 	%f33, %f32, %f32, %f29;
	max.f32 	%f34, %f100, %f33;
	add.s32 	%r41, %r66, %r22;
	mul.wide.s32 	%rd47, %r22, 4;
	add.s64 	%rd48, %rd36, %rd47;
	add.s64 	%rd49, %rd38, %rd47;
	ld.global.nc.f32 	%f35, [%rd49];
	ld.global.nc.f32 	%f36, [%rd48];
	sub.f32 	%f37, %f36, %f35;
	add.s64 	%rd50, %rd40, %rd47;
	add.s64 	%rd51, %rd42, %rd47;
	ld.global.nc.f32 	%f38, [%rd51];
	ld.global.nc.f32 	%f39, [%rd50];
	sub.f32 	%f40, %f39, %f38;
	mul.f32 	%f41, %f40, %f40;
	fma.rn.f32 	%f42, %f37, %f37, %f41;
	add.s64 	%rd52, %rd44, %rd47;
	add.s64 	%rd53, %rd46, %rd47;
	ld.global.nc.f32 	%f43, [%rd53];
	ld.global.nc.f32 	%f44, [%rd52];
	sub.f32 	%f45, %f44, %f43;
	fma.rn.f32 	%f46, %f45, %f45, %f42;
	max.f32 	%f47, %f34, %f46;
	add.s32 	%r42, %r41, %r22;
	add.s64 	%rd54, %rd48, %rd47;
	add.s64 	%rd55, %rd49, %rd47;
	ld.global.nc.f32 	%f48, [%rd55];
	ld.global.nc.f32 	%f49, [%rd54];
	sub.f32 	%f50, %f49, %f48;
	add.s64 	%rd56, %rd50, %rd47;
	add.s64 	%rd57, %rd51, %rd47;
	ld.global.nc.f32 	%f51, [%rd57];
	ld.global.nc.f32 	%f52, [%rd56];
	sub.f32 	%f53, %f52, %f51;
	mul.f32 	%f54, %f53, %f53;
	fma.rn.f32 	%f55, %f50, %f50, %f54;
	add.s64 	%rd58, %rd52, %rd47;
	add.s64 	%rd59, %rd53, %rd47;
	ld.global.nc.f32 	%f56, [%rd59];
	ld.global.nc.f32 	%f57, [%rd58];
	sub.f32 	%f58, %f57, %f56;
	fma.rn.f32 	%f59, %f58, %f58, %f55;
	max.f32 	%f60, %f47, %f59;
	add.s32 	%r43, %r42, %r22;
	add.s64 	%rd60, %rd54, %rd47;
	add.s64 	%rd61, %rd55, %rd47;
	ld.global.nc.f32 	%f61, [%rd61];
	ld.global.nc.f32 	%f62, [%rd60];
	sub.f32 	%f63, %f62, %f61;
	add.s64 	%rd62, %rd56, %rd47;
	add.s64 	%rd63, %rd57, %rd47;
	ld.global.nc.f32 	%f64, [%rd63];
	ld.global.nc.f32 	%f65, [%rd62];
	sub.f32 	%f66, %f65, %f64;
	mul.f32 	%f67, %f66, %f66;
	fma.rn.f32 	%f68, %f63, %f63, %f67;
	add.s64 	%rd64, %rd58, %rd47;
	add.s64 	%rd65, %rd59, %rd47;
	ld.global.nc.f32 	%f69, [%rd65];
	ld.global.nc.f32 	%f70, [%rd64];
	sub.f32 	%f71, %f70, %f69;
	fma.rn.f32 	%f72, %f71, %f71, %f68;
	max.f32 	%f100, %f60, %f72;
	add.s32 	%r66, %r43, %r22;
	setp.lt.s32 	%p5, %r66, %r15;
	@%p5 bra 	LBB0_6;

LBB0_7:
	shl.b32 	%r45, %r16, 2;
	mov.u32 	%r46, _ZZ17reducemaxvecdiff2E5sdata;
	add.s32 	%r47, %r46, %r45;
	st.shared.f32 	[%r47], %f100;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r68, 66;
	@%p6 bra 	LBB0_11;

LBB0_8:
	shr.u32 	%r14, %r68, 1;
	setp.ge.u32 	%p7, %r16, %r14;
	@%p7 bra 	LBB0_10;

	shl.b32 	%r54, %r14, 2;
	add.s32 	%r55, %r47, %r54;
	ld.shared.f32 	%f73, [%r55];
	ld.shared.f32 	%f74, [%r47];
	max.f32 	%f75, %f74, %f73;
	st.shared.f32 	[%r47], %f75;

LBB0_10:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r68, 131;
	mov.u32 	%r68, %r14;
	@%p8 bra 	LBB0_8;

LBB0_11:
	setp.gt.s32 	%p9, %r16, 31;
	@%p9 bra 	LBB0_13;

	ld.volatile.shared.f32 	%f76, [%r47+128];
	ld.volatile.shared.f32 	%f77, [%r47];
	max.f32 	%f78, %f77, %f76;
	st.volatile.shared.f32 	[%r47], %f78;
	ld.volatile.shared.f32 	%f79, [%r47+64];
	ld.volatile.shared.f32 	%f80, [%r47];
	max.f32 	%f81, %f80, %f79;
	st.volatile.shared.f32 	[%r47], %f81;
	ld.volatile.shared.f32 	%f82, [%r47+32];
	ld.volatile.shared.f32 	%f83, [%r47];
	max.f32 	%f84, %f83, %f82;
	st.volatile.shared.f32 	[%r47], %f84;
	ld.volatile.shared.f32 	%f85, [%r47+16];
	ld.volatile.shared.f32 	%f86, [%r47];
	max.f32 	%f87, %f86, %f85;
	st.volatile.shared.f32 	[%r47], %f87;
	ld.volatile.shared.f32 	%f88, [%r47+8];
	ld.volatile.shared.f32 	%f89, [%r47];
	max.f32 	%f90, %f89, %f88;
	st.volatile.shared.f32 	[%r47], %f90;
	ld.volatile.shared.f32 	%f91, [%r47+4];
	ld.volatile.shared.f32 	%f92, [%r47];
	max.f32 	%f93, %f92, %f91;
	st.volatile.shared.f32 	[%r47], %f93;

LBB0_13:
	setp.ne.s32 	%p10, %r16, 0;
	@%p10 bra 	LBB0_15;

	ld.shared.f32 	%f94, [_ZZ17reducemaxvecdiff2E5sdata];
	abs.f32 	%f95, %f94;
	cvta.to.global.u64 	%rd66, %rd26;
	mov.b32 	%r62, %f95;
	atom.global.max.s32 	%r63, [%rd66], %r62;

LBB0_15:
	ret;

}

`
)
