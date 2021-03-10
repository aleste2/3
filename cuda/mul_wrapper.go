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

// CUDA handle for mul kernel
var mul_code cu.Function

// Stores the arguments for mul kernel invocation
type mul_args_t struct {
	arg_dst unsafe.Pointer
	arg_a   unsafe.Pointer
	arg_b   unsafe.Pointer
	arg_N   int
	argptr  [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for mul kernel invocation
var mul_args mul_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	mul_args.argptr[0] = unsafe.Pointer(&mul_args.arg_dst)
	mul_args.argptr[1] = unsafe.Pointer(&mul_args.arg_a)
	mul_args.argptr[2] = unsafe.Pointer(&mul_args.arg_b)
	mul_args.argptr[3] = unsafe.Pointer(&mul_args.arg_N)
}

// Wrapper for mul CUDA kernel, asynchronous.
func k_mul_async(dst unsafe.Pointer, a unsafe.Pointer, b unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("mul")
	}

	mul_args.Lock()
	defer mul_args.Unlock()

	if mul_code == 0 {
		mul_code = fatbinLoad(mul_map, "mul")
	}

	mul_args.arg_dst = dst
	mul_args.arg_a = a
	mul_args.arg_b = b
	mul_args.arg_N = N

	args := mul_args.argptr[:]
	cu.LaunchKernel(mul_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("mul")
	}
}

// maps compute capability on PTX code for mul kernel.
var mul_map = map[int]string{0: "",
	70: mul_ptx_70}

// mul PTX code for various compute capabilities.
const (
	mul_ptx_70 = `
.version 7.2
.target sm_70
.address_size 64

	// .globl	mul

.visible .entry mul(
	.param .u64 mul_param_0,
	.param .u64 mul_param_1,
	.param .u64 mul_param_2,
	.param .u32 mul_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .f32 	%f<4>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<11>;


	ld.param.u64 	%rd1, [mul_param_0];
	ld.param.u64 	%rd2, [mul_param_1];
	ld.param.u64 	%rd3, [mul_param_2];
	ld.param.u32 	%r2, [mul_param_3];
	mov.u32 	%r3, %ctaid.y;
	mov.u32 	%r4, %nctaid.x;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	LBB0_2;

	cvta.to.global.u64 	%rd4, %rd2;
	mul.wide.s32 	%rd5, %r1, 4;
	add.s64 	%rd6, %rd4, %rd5;
	cvta.to.global.u64 	%rd7, %rd3;
	add.s64 	%rd8, %rd7, %rd5;
	ld.global.nc.f32 	%f1, [%rd8];
	ld.global.nc.f32 	%f2, [%rd6];
	mul.f32 	%f3, %f2, %f1;
	cvta.to.global.u64 	%rd9, %rd1;
	add.s64 	%rd10, %rd9, %rd5;
	st.global.f32 	[%rd10], %f3;

LBB0_2:
	ret;

}

`
)
