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

// CUDA handle for InittemperatureJH kernel
var InittemperatureJH_code cu.Function

// Stores the arguments for InittemperatureJH kernel invocation
type InittemperatureJH_args_t struct {
	arg_tempJH    unsafe.Pointer
	arg_TSubs_    unsafe.Pointer
	arg_TSubs_mul float32
	arg_N         int
	argptr        [4]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for InittemperatureJH kernel invocation
var InittemperatureJH_args InittemperatureJH_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	InittemperatureJH_args.argptr[0] = unsafe.Pointer(&InittemperatureJH_args.arg_tempJH)
	InittemperatureJH_args.argptr[1] = unsafe.Pointer(&InittemperatureJH_args.arg_TSubs_)
	InittemperatureJH_args.argptr[2] = unsafe.Pointer(&InittemperatureJH_args.arg_TSubs_mul)
	InittemperatureJH_args.argptr[3] = unsafe.Pointer(&InittemperatureJH_args.arg_N)
}

// Wrapper for InittemperatureJH CUDA kernel, asynchronous.
func k_InittemperatureJH_async(tempJH unsafe.Pointer, TSubs_ unsafe.Pointer, TSubs_mul float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("InittemperatureJH")
	}

	InittemperatureJH_args.Lock()
	defer InittemperatureJH_args.Unlock()

	if InittemperatureJH_code == 0 {
		InittemperatureJH_code = fatbinLoad(InittemperatureJH_map, "InittemperatureJH")
	}

	InittemperatureJH_args.arg_tempJH = tempJH
	InittemperatureJH_args.arg_TSubs_ = TSubs_
	InittemperatureJH_args.arg_TSubs_mul = TSubs_mul
	InittemperatureJH_args.arg_N = N

	args := InittemperatureJH_args.argptr[:]
	cu.LaunchKernel(InittemperatureJH_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("InittemperatureJH")
	}
}

// maps compute capability on PTX code for InittemperatureJH kernel.
var InittemperatureJH_map = map[int]string{0: "",
	30: InittemperatureJH_ptx_30}

// InittemperatureJH PTX code for various compute capabilities.
const (
	InittemperatureJH_ptx_30 = `
.version 6.5
.target sm_30
.address_size 64

	// .globl	InittemperatureJH

.visible .entry InittemperatureJH(
	.param .u64 InittemperatureJH_param_0,
	.param .u64 InittemperatureJH_param_1,
	.param .f32 InittemperatureJH_param_2,
	.param .u32 InittemperatureJH_param_3
)
{
	.reg .pred 	%p<3>;
	.reg .f32 	%f<6>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<9>;


	ld.param.u64 	%rd1, [InittemperatureJH_param_0];
	ld.param.u64 	%rd2, [InittemperatureJH_param_1];
	ld.param.f32 	%f5, [InittemperatureJH_param_2];
	ld.param.u32 	%r2, [InittemperatureJH_param_3];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_4;

	setp.eq.s64	%p2, %rd2, 0;
	@%p2 bra 	BB0_3;

	cvta.to.global.u64 	%rd3, %rd2;
	mul.wide.s32 	%rd4, %r1, 4;
	add.s64 	%rd5, %rd3, %rd4;
	ld.global.f32 	%f4, [%rd5];
	mul.f32 	%f5, %f4, %f5;

BB0_3:
	cvta.to.global.u64 	%rd6, %rd1;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	st.global.f32 	[%rd8], %f5;

BB0_4:
	ret;
}


`
)
