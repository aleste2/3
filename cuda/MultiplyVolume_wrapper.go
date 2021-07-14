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

// CUDA handle for MultiplyVolume kernel
var MultiplyVolume_code cu.Function

// Stores the arguments for MultiplyVolume kernel invocation
type MultiplyVolume_args_t struct {
	arg_Bx     unsafe.Pointer
	arg_By     unsafe.Pointer
	arg_Bz     unsafe.Pointer
	arg_volume float32
	arg_N      int
	argptr     [5]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for MultiplyVolume kernel invocation
var MultiplyVolume_args MultiplyVolume_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	MultiplyVolume_args.argptr[0] = unsafe.Pointer(&MultiplyVolume_args.arg_Bx)
	MultiplyVolume_args.argptr[1] = unsafe.Pointer(&MultiplyVolume_args.arg_By)
	MultiplyVolume_args.argptr[2] = unsafe.Pointer(&MultiplyVolume_args.arg_Bz)
	MultiplyVolume_args.argptr[3] = unsafe.Pointer(&MultiplyVolume_args.arg_volume)
	MultiplyVolume_args.argptr[4] = unsafe.Pointer(&MultiplyVolume_args.arg_N)
}

// Wrapper for MultiplyVolume CUDA kernel, asynchronous.
func k_MultiplyVolume_async(Bx unsafe.Pointer, By unsafe.Pointer, Bz unsafe.Pointer, volume float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("MultiplyVolume")
	}

	MultiplyVolume_args.Lock()
	defer MultiplyVolume_args.Unlock()

	if MultiplyVolume_code == 0 {
		MultiplyVolume_code = fatbinLoad(MultiplyVolume_map, "MultiplyVolume")
	}

	MultiplyVolume_args.arg_Bx = Bx
	MultiplyVolume_args.arg_By = By
	MultiplyVolume_args.arg_Bz = Bz
	MultiplyVolume_args.arg_volume = volume
	MultiplyVolume_args.arg_N = N

	args := MultiplyVolume_args.argptr[:]
	cu.LaunchKernel(MultiplyVolume_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("MultiplyVolume")
	}
}

// maps compute capability on PTX code for MultiplyVolume kernel.
var MultiplyVolume_map = map[int]string{0: "",
	70: MultiplyVolume_ptx_70}

// MultiplyVolume PTX code for various compute capabilities.
const (
	MultiplyVolume_ptx_70 = `
.version 7.1
.target sm_70
.address_size 64

	// .globl	MultiplyVolume

.visible .entry MultiplyVolume(
	.param .u64 MultiplyVolume_param_0,
	.param .u64 MultiplyVolume_param_1,
	.param .u64 MultiplyVolume_param_2,
	.param .f32 MultiplyVolume_param_3,
	.param .u32 MultiplyVolume_param_4
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<8>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<11>;


	ld.param.u64 	%rd1, [MultiplyVolume_param_0];
	ld.param.u64 	%rd2, [MultiplyVolume_param_1];
	ld.param.u64 	%rd3, [MultiplyVolume_param_2];
	ld.param.f32 	%f1, [MultiplyVolume_param_3];
	ld.param.u32 	%r2, [MultiplyVolume_param_4];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd4, %rd1;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	ld.global.f32 	%f2, [%rd6];
	mul.f32 	%f3, %f2, %f1;
	st.global.f32 	[%rd6], %f3;
	cvta.to.global.u64 	%rd7, %rd2;
	add.s64 	%rd8, %rd7, %rd5;
	ld.global.f32 	%f4, [%rd8];
	mul.f32 	%f5, %f4, %f1;
	st.global.f32 	[%rd8], %f5;
	cvta.to.global.u64 	%rd9, %rd3;
	add.s64 	%rd10, %rd9, %rd5;
	ld.global.f32 	%f6, [%rd10];
	mul.f32 	%f7, %f6, %f1;
	st.global.f32 	[%rd10], %f7;

BB0_2:
	ret;
}


`
)
