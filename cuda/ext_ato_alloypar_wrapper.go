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

// CUDA handle for alloyparcuda kernel
var alloyparcuda_code cu.Function

// Stores the arguments for alloyparcuda kernel invocation
type alloyparcuda_args_t struct {
	arg_host    byte
	arg_alloy   byte
	arg_percent float32
	arg_random  unsafe.Pointer
	arg_regions unsafe.Pointer
	arg_N       int
	argptr      [6]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for alloyparcuda kernel invocation
var alloyparcuda_args alloyparcuda_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	alloyparcuda_args.argptr[0] = unsafe.Pointer(&alloyparcuda_args.arg_host)
	alloyparcuda_args.argptr[1] = unsafe.Pointer(&alloyparcuda_args.arg_alloy)
	alloyparcuda_args.argptr[2] = unsafe.Pointer(&alloyparcuda_args.arg_percent)
	alloyparcuda_args.argptr[3] = unsafe.Pointer(&alloyparcuda_args.arg_random)
	alloyparcuda_args.argptr[4] = unsafe.Pointer(&alloyparcuda_args.arg_regions)
	alloyparcuda_args.argptr[5] = unsafe.Pointer(&alloyparcuda_args.arg_N)
}

// Wrapper for alloyparcuda CUDA kernel, asynchronous.
func k_alloyparcuda_async(host byte, alloy byte, percent float32, random unsafe.Pointer, regions unsafe.Pointer, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("alloyparcuda")
	}

	alloyparcuda_args.Lock()
	defer alloyparcuda_args.Unlock()

	if alloyparcuda_code == 0 {
		alloyparcuda_code = fatbinLoad(alloyparcuda_map, "alloyparcuda")
	}

	alloyparcuda_args.arg_host = host
	alloyparcuda_args.arg_alloy = alloy
	alloyparcuda_args.arg_percent = percent
	alloyparcuda_args.arg_random = random
	alloyparcuda_args.arg_regions = regions
	alloyparcuda_args.arg_N = N

	args := alloyparcuda_args.argptr[:]
	cu.LaunchKernel(alloyparcuda_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("alloyparcuda")
	}
}

// maps compute capability on PTX code for alloyparcuda kernel.
var alloyparcuda_map = map[int]string{0: "",
	70: alloyparcuda_ptx_70}

// alloyparcuda PTX code for various compute capabilities.
const (
	alloyparcuda_ptx_70 = `
.version 7.1
.target sm_70
.address_size 64

	// .globl	alloyparcuda

.visible .entry alloyparcuda(
	.param .u8 alloyparcuda_param_0,
	.param .u8 alloyparcuda_param_1,
	.param .f32 alloyparcuda_param_2,
	.param .u64 alloyparcuda_param_3,
	.param .u64 alloyparcuda_param_4,
	.param .u32 alloyparcuda_param_5
)
{
	.reg .pred 	%p<4>;
	.reg .b16 	%rs<4>;
	.reg .f32 	%f<3>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<9>;


	ld.param.f32 	%f1, [alloyparcuda_param_2];
	ld.param.u64 	%rd2, [alloyparcuda_param_3];
	ld.param.u64 	%rd3, [alloyparcuda_param_4];
	ld.param.u32 	%r2, [alloyparcuda_param_5];
	ld.param.u8 	%rs2, [alloyparcuda_param_1];
	ld.param.u8 	%rs1, [alloyparcuda_param_0];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_4;

	cvta.to.global.u64 	%rd4, %rd3;
	cvt.s64.s32	%rd5, %r1;
	add.s64 	%rd1, %rd4, %rd5;
	ld.global.u8 	%rs3, [%rd1];
	setp.ne.s16	%p2, %rs3, %rs1;
	@%p2 bra 	BB0_4;

	cvta.to.global.u64 	%rd6, %rd2;
	mul.wide.s32 	%rd7, %r1, 4;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.f32 	%f2, [%rd8];
	setp.gtu.f32	%p3, %f2, %f1;
	@%p3 bra 	BB0_4;

	st.global.u8 	[%rd1], %rs2;

BB0_4:
	ret;
}


`
)
