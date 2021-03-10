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

// CUDA handle for reducedot kernel
var reducedot_code cu.Function

// Stores the arguments for reducedot kernel invocation
type reducedot_args_t struct {
	arg_x1      unsafe.Pointer
	arg_x2      unsafe.Pointer
	arg_dst     unsafe.Pointer
	arg_initVal float32
	arg_n       int
	argptr      [5]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for reducedot kernel invocation
var reducedot_args reducedot_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	reducedot_args.argptr[0] = unsafe.Pointer(&reducedot_args.arg_x1)
	reducedot_args.argptr[1] = unsafe.Pointer(&reducedot_args.arg_x2)
	reducedot_args.argptr[2] = unsafe.Pointer(&reducedot_args.arg_dst)
	reducedot_args.argptr[3] = unsafe.Pointer(&reducedot_args.arg_initVal)
	reducedot_args.argptr[4] = unsafe.Pointer(&reducedot_args.arg_n)
}

// Wrapper for reducedot CUDA kernel, asynchronous.
func k_reducedot_async(x1 unsafe.Pointer, x2 unsafe.Pointer, dst unsafe.Pointer, initVal float32, n int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("reducedot")
	}

	reducedot_args.Lock()
	defer reducedot_args.Unlock()

	if reducedot_code == 0 {
		reducedot_code = fatbinLoad(reducedot_map, "reducedot")
	}

	reducedot_args.arg_x1 = x1
	reducedot_args.arg_x2 = x2
	reducedot_args.arg_dst = dst
	reducedot_args.arg_initVal = initVal
	reducedot_args.arg_n = n

	args := reducedot_args.argptr[:]
	cu.LaunchKernel(reducedot_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("reducedot")
	}
}

// maps compute capability on PTX code for reducedot kernel.
var reducedot_map = map[int]string{0: "",
	70: reducedot_ptx_70}

// reducedot PTX code for various compute capabilities.
const (
	reducedot_ptx_70 = `
.version 7.2
.target sm_70
.address_size 64

	// .globl	reducedot

.visible .entry reducedot(
	.param .u64 reducedot_param_0,
	.param .u64 reducedot_param_1,
	.param .u64 reducedot_param_2,
	.param .f32 reducedot_param_3,
	.param .u32 reducedot_param_4
)
{
	.reg .pred 	%p<11>;
	.reg .f32 	%f<51>;
	.reg .b32 	%r<54>;
	.reg .b64 	%rd<28>;
	// demoted variable
	.shared .align 4 .b8 _ZZ9reducedotE5sdata[2048];

	ld.param.u64 	%rd13, [reducedot_param_0];
	ld.param.u64 	%rd11, [reducedot_param_1];
	ld.param.u64 	%rd12, [reducedot_param_2];
	ld.param.f32 	%f50, [reducedot_param_3];
	ld.param.u32 	%r18, [reducedot_param_4];
	cvta.to.global.u64 	%rd1, %rd13;
	mov.u32 	%r53, %ntid.x;
	mov.u32 	%r20, %ctaid.x;
	mov.u32 	%r1, %tid.x;
	mad.lo.s32 	%r21, %r20, %r53, %r1;
	setp.ge.s32 	%p1, %r21, %r18;
	@%p1 bra 	LBB0_7;

	mov.u32 	%r22, %nctaid.x;
	mov.u32 	%r23, %ntid.x;
	mul.lo.s32 	%r2, %r22, %r23;
	add.s32 	%r24, %r2, %r18;
	mad.lo.s32 	%r51, %r20, %r23, %r1;
	add.s32 	%r27, %r51, %r2;
	not.b32 	%r28, %r27;
	add.s32 	%r29, %r24, %r28;
	div.u32 	%r4, %r29, %r2;
	add.s32 	%r30, %r4, 1;
	and.b32  	%r50, %r30, 3;
	setp.eq.s32 	%p2, %r50, 0;
	@%p2 bra 	LBB0_4;

	mad.lo.s32 	%r51, %r20, %r23, %r1;
	cvta.to.global.u64 	%rd14, %rd11;
	mul.wide.s32 	%rd15, %r51, 4;
	add.s64 	%rd27, %rd14, %rd15;
	mul.wide.s32 	%rd3, %r2, 4;
	add.s64 	%rd26, %rd1, %rd15;

LBB0_3:
	.pragma "nounroll";
	ld.global.nc.f32 	%f10, [%rd27];
	ld.global.nc.f32 	%f11, [%rd26];
	fma.rn.f32 	%f50, %f11, %f10, %f50;
	add.s32 	%r51, %r51, %r2;
	add.s64 	%rd27, %rd27, %rd3;
	add.s64 	%rd26, %rd26, %rd3;
	add.s32 	%r50, %r50, -1;
	setp.ne.s32 	%p3, %r50, 0;
	@%p3 bra 	LBB0_3;

LBB0_4:
	setp.lt.u32 	%p4, %r4, 3;
	@%p4 bra 	LBB0_7;

	mul.wide.s32 	%rd9, %r2, 4;
	cvta.to.global.u64 	%rd10, %rd11;

LBB0_6:
	mul.wide.s32 	%rd16, %r51, 4;
	add.s64 	%rd17, %rd1, %rd16;
	add.s64 	%rd18, %rd10, %rd16;
	ld.global.nc.f32 	%f12, [%rd18];
	ld.global.nc.f32 	%f13, [%rd17];
	fma.rn.f32 	%f14, %f13, %f12, %f50;
	add.s64 	%rd19, %rd17, %rd9;
	add.s64 	%rd20, %rd18, %rd9;
	ld.global.nc.f32 	%f15, [%rd20];
	ld.global.nc.f32 	%f16, [%rd19];
	fma.rn.f32 	%f17, %f16, %f15, %f14;
	add.s32 	%r41, %r51, %r2;
	add.s32 	%r42, %r41, %r2;
	add.s64 	%rd21, %rd19, %rd9;
	add.s64 	%rd22, %rd20, %rd9;
	ld.global.nc.f32 	%f18, [%rd22];
	ld.global.nc.f32 	%f19, [%rd21];
	fma.rn.f32 	%f20, %f19, %f18, %f17;
	add.s32 	%r43, %r42, %r2;
	add.s64 	%rd23, %rd21, %rd9;
	add.s64 	%rd24, %rd22, %rd9;
	ld.global.nc.f32 	%f21, [%rd24];
	ld.global.nc.f32 	%f22, [%rd23];
	fma.rn.f32 	%f50, %f22, %f21, %f20;
	add.s32 	%r51, %r43, %r2;
	setp.lt.s32 	%p5, %r51, %r18;
	@%p5 bra 	LBB0_6;

LBB0_7:
	shl.b32 	%r44, %r1, 2;
	mov.u32 	%r45, _ZZ9reducedotE5sdata;
	add.s32 	%r14, %r45, %r44;
	st.shared.f32 	[%r14], %f50;
	bar.sync 	0;
	setp.lt.u32 	%p6, %r53, 66;
	@%p6 bra 	LBB0_11;

LBB0_8:
	shr.u32 	%r17, %r53, 1;
	setp.ge.u32 	%p7, %r1, %r17;
	@%p7 bra 	LBB0_10;

	ld.shared.f32 	%f23, [%r14];
	shl.b32 	%r47, %r17, 2;
	add.s32 	%r48, %r14, %r47;
	ld.shared.f32 	%f24, [%r48];
	add.f32 	%f25, %f23, %f24;
	st.shared.f32 	[%r14], %f25;

LBB0_10:
	bar.sync 	0;
	setp.gt.u32 	%p8, %r53, 131;
	mov.u32 	%r53, %r17;
	@%p8 bra 	LBB0_8;

LBB0_11:
	setp.gt.s32 	%p9, %r1, 31;
	@%p9 bra 	LBB0_13;

	ld.volatile.shared.f32 	%f26, [%r14];
	ld.volatile.shared.f32 	%f27, [%r14+128];
	add.f32 	%f28, %f26, %f27;
	st.volatile.shared.f32 	[%r14], %f28;
	ld.volatile.shared.f32 	%f29, [%r14+64];
	ld.volatile.shared.f32 	%f30, [%r14];
	add.f32 	%f31, %f30, %f29;
	st.volatile.shared.f32 	[%r14], %f31;
	ld.volatile.shared.f32 	%f32, [%r14+32];
	ld.volatile.shared.f32 	%f33, [%r14];
	add.f32 	%f34, %f33, %f32;
	st.volatile.shared.f32 	[%r14], %f34;
	ld.volatile.shared.f32 	%f35, [%r14+16];
	ld.volatile.shared.f32 	%f36, [%r14];
	add.f32 	%f37, %f36, %f35;
	st.volatile.shared.f32 	[%r14], %f37;
	ld.volatile.shared.f32 	%f38, [%r14+8];
	ld.volatile.shared.f32 	%f39, [%r14];
	add.f32 	%f40, %f39, %f38;
	st.volatile.shared.f32 	[%r14], %f40;
	ld.volatile.shared.f32 	%f41, [%r14+4];
	ld.volatile.shared.f32 	%f42, [%r14];
	add.f32 	%f43, %f42, %f41;
	st.volatile.shared.f32 	[%r14], %f43;

LBB0_13:
	setp.ne.s32 	%p10, %r1, 0;
	@%p10 bra 	LBB0_15;

	ld.shared.f32 	%f44, [_ZZ9reducedotE5sdata];
	cvta.to.global.u64 	%rd25, %rd12;
	atom.global.add.f32 	%f45, [%rd25], %f44;

LBB0_15:
	ret;

}

`
)
