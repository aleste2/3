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

// CUDA handle for adduniaxialanisotropyato kernel
var adduniaxialanisotropyato_code cu.Function

// Stores the arguments for adduniaxialanisotropyato kernel invocation
type adduniaxialanisotropyato_args_t struct {
	arg_Bx       unsafe.Pointer
	arg_By       unsafe.Pointer
	arg_Bz       unsafe.Pointer
	arg_mx       unsafe.Pointer
	arg_my       unsafe.Pointer
	arg_mz       unsafe.Pointer
	arg_Mu_      unsafe.Pointer
	arg_Mu_mul   float32
	arg_Dato_    unsafe.Pointer
	arg_Dato_mul float32
	arg_ux_      unsafe.Pointer
	arg_ux_mul   float32
	arg_uy_      unsafe.Pointer
	arg_uy_mul   float32
	arg_uz_      unsafe.Pointer
	arg_uz_mul   float32
	arg_N        int
	argptr       [17]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for adduniaxialanisotropyato kernel invocation
var adduniaxialanisotropyato_args adduniaxialanisotropyato_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	adduniaxialanisotropyato_args.argptr[0] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_Bx)
	adduniaxialanisotropyato_args.argptr[1] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_By)
	adduniaxialanisotropyato_args.argptr[2] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_Bz)
	adduniaxialanisotropyato_args.argptr[3] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_mx)
	adduniaxialanisotropyato_args.argptr[4] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_my)
	adduniaxialanisotropyato_args.argptr[5] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_mz)
	adduniaxialanisotropyato_args.argptr[6] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_Mu_)
	adduniaxialanisotropyato_args.argptr[7] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_Mu_mul)
	adduniaxialanisotropyato_args.argptr[8] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_Dato_)
	adduniaxialanisotropyato_args.argptr[9] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_Dato_mul)
	adduniaxialanisotropyato_args.argptr[10] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_ux_)
	adduniaxialanisotropyato_args.argptr[11] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_ux_mul)
	adduniaxialanisotropyato_args.argptr[12] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_uy_)
	adduniaxialanisotropyato_args.argptr[13] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_uy_mul)
	adduniaxialanisotropyato_args.argptr[14] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_uz_)
	adduniaxialanisotropyato_args.argptr[15] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_uz_mul)
	adduniaxialanisotropyato_args.argptr[16] = unsafe.Pointer(&adduniaxialanisotropyato_args.arg_N)
}

// Wrapper for adduniaxialanisotropyato CUDA kernel, asynchronous.
func k_adduniaxialanisotropyato_async(Bx unsafe.Pointer, By unsafe.Pointer, Bz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Mu_ unsafe.Pointer, Mu_mul float32, Dato_ unsafe.Pointer, Dato_mul float32, ux_ unsafe.Pointer, ux_mul float32, uy_ unsafe.Pointer, uy_mul float32, uz_ unsafe.Pointer, uz_mul float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("adduniaxialanisotropyato")
	}

	adduniaxialanisotropyato_args.Lock()
	defer adduniaxialanisotropyato_args.Unlock()

	if adduniaxialanisotropyato_code == 0 {
		adduniaxialanisotropyato_code = fatbinLoad(adduniaxialanisotropyato_map, "adduniaxialanisotropyato")
	}

	adduniaxialanisotropyato_args.arg_Bx = Bx
	adduniaxialanisotropyato_args.arg_By = By
	adduniaxialanisotropyato_args.arg_Bz = Bz
	adduniaxialanisotropyato_args.arg_mx = mx
	adduniaxialanisotropyato_args.arg_my = my
	adduniaxialanisotropyato_args.arg_mz = mz
	adduniaxialanisotropyato_args.arg_Mu_ = Mu_
	adduniaxialanisotropyato_args.arg_Mu_mul = Mu_mul
	adduniaxialanisotropyato_args.arg_Dato_ = Dato_
	adduniaxialanisotropyato_args.arg_Dato_mul = Dato_mul
	adduniaxialanisotropyato_args.arg_ux_ = ux_
	adduniaxialanisotropyato_args.arg_ux_mul = ux_mul
	adduniaxialanisotropyato_args.arg_uy_ = uy_
	adduniaxialanisotropyato_args.arg_uy_mul = uy_mul
	adduniaxialanisotropyato_args.arg_uz_ = uz_
	adduniaxialanisotropyato_args.arg_uz_mul = uz_mul
	adduniaxialanisotropyato_args.arg_N = N

	args := adduniaxialanisotropyato_args.argptr[:]
	cu.LaunchKernel(adduniaxialanisotropyato_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("adduniaxialanisotropyato")
	}
}

// maps compute capability on PTX code for adduniaxialanisotropyato kernel.
var adduniaxialanisotropyato_map = map[int]string{0: "",
	70: adduniaxialanisotropyato_ptx_70}

// adduniaxialanisotropyato PTX code for various compute capabilities.
const (
	adduniaxialanisotropyato_ptx_70 = `
.version 7.2
.target sm_70
.address_size 64

	// .globl	adduniaxialanisotropyato

.visible .entry adduniaxialanisotropyato(
	.param .u64 adduniaxialanisotropyato_param_0,
	.param .u64 adduniaxialanisotropyato_param_1,
	.param .u64 adduniaxialanisotropyato_param_2,
	.param .u64 adduniaxialanisotropyato_param_3,
	.param .u64 adduniaxialanisotropyato_param_4,
	.param .u64 adduniaxialanisotropyato_param_5,
	.param .u64 adduniaxialanisotropyato_param_6,
	.param .f32 adduniaxialanisotropyato_param_7,
	.param .u64 adduniaxialanisotropyato_param_8,
	.param .f32 adduniaxialanisotropyato_param_9,
	.param .u64 adduniaxialanisotropyato_param_10,
	.param .f32 adduniaxialanisotropyato_param_11,
	.param .u64 adduniaxialanisotropyato_param_12,
	.param .f32 adduniaxialanisotropyato_param_13,
	.param .u64 adduniaxialanisotropyato_param_14,
	.param .f32 adduniaxialanisotropyato_param_15,
	.param .u32 adduniaxialanisotropyato_param_16
)
{
	.reg .pred 	%p<9>;
	.reg .f32 	%f<56>;
	.reg .b32 	%r<9>;
	.reg .b64 	%rd<40>;


	ld.param.u64 	%rd1, [adduniaxialanisotropyato_param_0];
	ld.param.u64 	%rd2, [adduniaxialanisotropyato_param_1];
	ld.param.u64 	%rd3, [adduniaxialanisotropyato_param_2];
	ld.param.u64 	%rd4, [adduniaxialanisotropyato_param_3];
	ld.param.u64 	%rd5, [adduniaxialanisotropyato_param_4];
	ld.param.u64 	%rd6, [adduniaxialanisotropyato_param_5];
	ld.param.u64 	%rd7, [adduniaxialanisotropyato_param_6];
	ld.param.f32 	%f53, [adduniaxialanisotropyato_param_7];
	ld.param.u64 	%rd8, [adduniaxialanisotropyato_param_8];
	ld.param.f32 	%f55, [adduniaxialanisotropyato_param_9];
	ld.param.u64 	%rd9, [adduniaxialanisotropyato_param_10];
	ld.param.f32 	%f49, [adduniaxialanisotropyato_param_11];
	ld.param.u64 	%rd10, [adduniaxialanisotropyato_param_12];
	ld.param.f32 	%f50, [adduniaxialanisotropyato_param_13];
	ld.param.u64 	%rd11, [adduniaxialanisotropyato_param_14];
	ld.param.f32 	%f51, [adduniaxialanisotropyato_param_15];
	ld.param.u32 	%r2, [adduniaxialanisotropyato_param_16];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	LBB0_16;

	setp.eq.s64 	%p2, %rd9, 0;
	@%p2 bra 	LBB0_3;

	cvta.to.global.u64 	%rd12, %rd9;
	mul.wide.s32 	%rd13, %r1, 4;
	add.s64 	%rd14, %rd12, %rd13;
	ld.global.nc.f32 	%f24, [%rd14];
	mul.f32 	%f49, %f24, %f49;

LBB0_3:
	setp.eq.s64 	%p3, %rd10, 0;
	@%p3 bra 	LBB0_5;

	cvta.to.global.u64 	%rd15, %rd10;
	mul.wide.s32 	%rd16, %r1, 4;
	add.s64 	%rd17, %rd15, %rd16;
	ld.global.nc.f32 	%f25, [%rd17];
	mul.f32 	%f50, %f25, %f50;

LBB0_5:
	setp.eq.s64 	%p4, %rd11, 0;
	@%p4 bra 	LBB0_7;

	cvta.to.global.u64 	%rd18, %rd11;
	mul.wide.s32 	%rd19, %r1, 4;
	add.s64 	%rd20, %rd18, %rd19;
	ld.global.nc.f32 	%f26, [%rd20];
	mul.f32 	%f51, %f26, %f51;

LBB0_7:
	mul.f32 	%f28, %f50, %f50;
	fma.rn.f32 	%f29, %f49, %f49, %f28;
	fma.rn.f32 	%f30, %f51, %f51, %f29;
	sqrt.rn.f32 	%f7, %f30;
	setp.eq.f32 	%p5, %f7, 0f00000000;
	mov.f32 	%f52, 0f00000000;
	@%p5 bra 	LBB0_9;

	rcp.rn.f32 	%f52, %f7;

LBB0_9:
	mul.f32 	%f10, %f49, %f52;
	mul.f32 	%f11, %f50, %f52;
	mul.f32 	%f12, %f51, %f52;
	setp.eq.s64 	%p6, %rd7, 0;
	@%p6 bra 	LBB0_11;

	cvta.to.global.u64 	%rd21, %rd7;
	mul.wide.s32 	%rd22, %r1, 4;
	add.s64 	%rd23, %rd21, %rd22;
	ld.global.nc.f32 	%f31, [%rd23];
	mul.f32 	%f53, %f31, %f53;

LBB0_11:
	setp.eq.f32 	%p7, %f53, 0f00000000;
	mov.f32 	%f54, 0f00000000;
	@%p7 bra 	LBB0_13;

	rcp.rn.f32 	%f54, %f53;

LBB0_13:
	setp.eq.s64 	%p8, %rd8, 0;
	@%p8 bra 	LBB0_15;

	cvta.to.global.u64 	%rd24, %rd8;
	mul.wide.s32 	%rd25, %r1, 4;
	add.s64 	%rd26, %rd24, %rd25;
	ld.global.nc.f32 	%f33, [%rd26];
	mul.f32 	%f55, %f33, %f55;

LBB0_15:
	cvta.to.global.u64 	%rd27, %rd5;
	mul.wide.s32 	%rd28, %r1, 4;
	add.s64 	%rd29, %rd27, %rd28;
	cvta.to.global.u64 	%rd30, %rd6;
	add.s64 	%rd31, %rd30, %rd28;
	cvta.to.global.u64 	%rd32, %rd4;
	add.s64 	%rd33, %rd32, %rd28;
	ld.global.nc.f32 	%f34, [%rd33];
	ld.global.nc.f32 	%f35, [%rd29];
	mul.f32 	%f36, %f11, %f35;
	fma.rn.f32 	%f37, %f10, %f34, %f36;
	ld.global.nc.f32 	%f38, [%rd31];
	fma.rn.f32 	%f39, %f12, %f38, %f37;
	mul.f32 	%f40, %f54, %f55;
	fma.rn.f32 	%f41, %f54, %f55, %f40;
	mul.f32 	%f42, %f41, %f39;
	cvta.to.global.u64 	%rd34, %rd1;
	add.s64 	%rd35, %rd34, %rd28;
	ld.global.f32 	%f43, [%rd35];
	fma.rn.f32 	%f44, %f10, %f42, %f43;
	st.global.f32 	[%rd35], %f44;
	cvta.to.global.u64 	%rd36, %rd2;
	add.s64 	%rd37, %rd36, %rd28;
	ld.global.f32 	%f45, [%rd37];
	fma.rn.f32 	%f46, %f11, %f42, %f45;
	st.global.f32 	[%rd37], %f46;
	cvta.to.global.u64 	%rd38, %rd3;
	add.s64 	%rd39, %rd38, %rd28;
	ld.global.f32 	%f47, [%rd39];
	fma.rn.f32 	%f48, %f12, %f42, %f47;
	st.global.f32 	[%rd39], %f48;

LBB0_16:
	ret;

}

`
)
