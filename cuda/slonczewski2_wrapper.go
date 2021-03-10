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

// CUDA handle for addslonczewskitorque2 kernel
var addslonczewskitorque2_code cu.Function

// Stores the arguments for addslonczewskitorque2 kernel invocation
type addslonczewskitorque2_args_t struct {
	arg_tx                unsafe.Pointer
	arg_ty                unsafe.Pointer
	arg_tz                unsafe.Pointer
	arg_mx                unsafe.Pointer
	arg_my                unsafe.Pointer
	arg_mz                unsafe.Pointer
	arg_Ms_               unsafe.Pointer
	arg_Ms_mul            float32
	arg_jz_               unsafe.Pointer
	arg_jz_mul            float32
	arg_px_               unsafe.Pointer
	arg_px_mul            float32
	arg_py_               unsafe.Pointer
	arg_py_mul            float32
	arg_pz_               unsafe.Pointer
	arg_pz_mul            float32
	arg_alpha_            unsafe.Pointer
	arg_alpha_mul         float32
	arg_pol_              unsafe.Pointer
	arg_pol_mul           float32
	arg_lambda_           unsafe.Pointer
	arg_lambda_mul        float32
	arg_epsPrime_         unsafe.Pointer
	arg_epsPrime_mul      float32
	arg_thickness_        unsafe.Pointer
	arg_thickness_mul     float32
	arg_meshThickness     float32
	arg_freeLayerPosition float32
	arg_N                 int
	argptr                [29]unsafe.Pointer
	sync.Mutex
}

// Stores the arguments for addslonczewskitorque2 kernel invocation
var addslonczewskitorque2_args addslonczewskitorque2_args_t

func init() {
	// CUDA driver kernel call wants pointers to arguments, set them up once.
	addslonczewskitorque2_args.argptr[0] = unsafe.Pointer(&addslonczewskitorque2_args.arg_tx)
	addslonczewskitorque2_args.argptr[1] = unsafe.Pointer(&addslonczewskitorque2_args.arg_ty)
	addslonczewskitorque2_args.argptr[2] = unsafe.Pointer(&addslonczewskitorque2_args.arg_tz)
	addslonczewskitorque2_args.argptr[3] = unsafe.Pointer(&addslonczewskitorque2_args.arg_mx)
	addslonczewskitorque2_args.argptr[4] = unsafe.Pointer(&addslonczewskitorque2_args.arg_my)
	addslonczewskitorque2_args.argptr[5] = unsafe.Pointer(&addslonczewskitorque2_args.arg_mz)
	addslonczewskitorque2_args.argptr[6] = unsafe.Pointer(&addslonczewskitorque2_args.arg_Ms_)
	addslonczewskitorque2_args.argptr[7] = unsafe.Pointer(&addslonczewskitorque2_args.arg_Ms_mul)
	addslonczewskitorque2_args.argptr[8] = unsafe.Pointer(&addslonczewskitorque2_args.arg_jz_)
	addslonczewskitorque2_args.argptr[9] = unsafe.Pointer(&addslonczewskitorque2_args.arg_jz_mul)
	addslonczewskitorque2_args.argptr[10] = unsafe.Pointer(&addslonczewskitorque2_args.arg_px_)
	addslonczewskitorque2_args.argptr[11] = unsafe.Pointer(&addslonczewskitorque2_args.arg_px_mul)
	addslonczewskitorque2_args.argptr[12] = unsafe.Pointer(&addslonczewskitorque2_args.arg_py_)
	addslonczewskitorque2_args.argptr[13] = unsafe.Pointer(&addslonczewskitorque2_args.arg_py_mul)
	addslonczewskitorque2_args.argptr[14] = unsafe.Pointer(&addslonczewskitorque2_args.arg_pz_)
	addslonczewskitorque2_args.argptr[15] = unsafe.Pointer(&addslonczewskitorque2_args.arg_pz_mul)
	addslonczewskitorque2_args.argptr[16] = unsafe.Pointer(&addslonczewskitorque2_args.arg_alpha_)
	addslonczewskitorque2_args.argptr[17] = unsafe.Pointer(&addslonczewskitorque2_args.arg_alpha_mul)
	addslonczewskitorque2_args.argptr[18] = unsafe.Pointer(&addslonczewskitorque2_args.arg_pol_)
	addslonczewskitorque2_args.argptr[19] = unsafe.Pointer(&addslonczewskitorque2_args.arg_pol_mul)
	addslonczewskitorque2_args.argptr[20] = unsafe.Pointer(&addslonczewskitorque2_args.arg_lambda_)
	addslonczewskitorque2_args.argptr[21] = unsafe.Pointer(&addslonczewskitorque2_args.arg_lambda_mul)
	addslonczewskitorque2_args.argptr[22] = unsafe.Pointer(&addslonczewskitorque2_args.arg_epsPrime_)
	addslonczewskitorque2_args.argptr[23] = unsafe.Pointer(&addslonczewskitorque2_args.arg_epsPrime_mul)
	addslonczewskitorque2_args.argptr[24] = unsafe.Pointer(&addslonczewskitorque2_args.arg_thickness_)
	addslonczewskitorque2_args.argptr[25] = unsafe.Pointer(&addslonczewskitorque2_args.arg_thickness_mul)
	addslonczewskitorque2_args.argptr[26] = unsafe.Pointer(&addslonczewskitorque2_args.arg_meshThickness)
	addslonczewskitorque2_args.argptr[27] = unsafe.Pointer(&addslonczewskitorque2_args.arg_freeLayerPosition)
	addslonczewskitorque2_args.argptr[28] = unsafe.Pointer(&addslonczewskitorque2_args.arg_N)
}

// Wrapper for addslonczewskitorque2 CUDA kernel, asynchronous.
func k_addslonczewskitorque2_async(tx unsafe.Pointer, ty unsafe.Pointer, tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, Ms_ unsafe.Pointer, Ms_mul float32, jz_ unsafe.Pointer, jz_mul float32, px_ unsafe.Pointer, px_mul float32, py_ unsafe.Pointer, py_mul float32, pz_ unsafe.Pointer, pz_mul float32, alpha_ unsafe.Pointer, alpha_mul float32, pol_ unsafe.Pointer, pol_mul float32, lambda_ unsafe.Pointer, lambda_mul float32, epsPrime_ unsafe.Pointer, epsPrime_mul float32, thickness_ unsafe.Pointer, thickness_mul float32, meshThickness float32, freeLayerPosition float32, N int, cfg *config) {
	if Synchronous { // debug
		Sync()
		timer.Start("addslonczewskitorque2")
	}

	addslonczewskitorque2_args.Lock()
	defer addslonczewskitorque2_args.Unlock()

	if addslonczewskitorque2_code == 0 {
		addslonczewskitorque2_code = fatbinLoad(addslonczewskitorque2_map, "addslonczewskitorque2")
	}

	addslonczewskitorque2_args.arg_tx = tx
	addslonczewskitorque2_args.arg_ty = ty
	addslonczewskitorque2_args.arg_tz = tz
	addslonczewskitorque2_args.arg_mx = mx
	addslonczewskitorque2_args.arg_my = my
	addslonczewskitorque2_args.arg_mz = mz
	addslonczewskitorque2_args.arg_Ms_ = Ms_
	addslonczewskitorque2_args.arg_Ms_mul = Ms_mul
	addslonczewskitorque2_args.arg_jz_ = jz_
	addslonczewskitorque2_args.arg_jz_mul = jz_mul
	addslonczewskitorque2_args.arg_px_ = px_
	addslonczewskitorque2_args.arg_px_mul = px_mul
	addslonczewskitorque2_args.arg_py_ = py_
	addslonczewskitorque2_args.arg_py_mul = py_mul
	addslonczewskitorque2_args.arg_pz_ = pz_
	addslonczewskitorque2_args.arg_pz_mul = pz_mul
	addslonczewskitorque2_args.arg_alpha_ = alpha_
	addslonczewskitorque2_args.arg_alpha_mul = alpha_mul
	addslonczewskitorque2_args.arg_pol_ = pol_
	addslonczewskitorque2_args.arg_pol_mul = pol_mul
	addslonczewskitorque2_args.arg_lambda_ = lambda_
	addslonczewskitorque2_args.arg_lambda_mul = lambda_mul
	addslonczewskitorque2_args.arg_epsPrime_ = epsPrime_
	addslonczewskitorque2_args.arg_epsPrime_mul = epsPrime_mul
	addslonczewskitorque2_args.arg_thickness_ = thickness_
	addslonczewskitorque2_args.arg_thickness_mul = thickness_mul
	addslonczewskitorque2_args.arg_meshThickness = meshThickness
	addslonczewskitorque2_args.arg_freeLayerPosition = freeLayerPosition
	addslonczewskitorque2_args.arg_N = N

	args := addslonczewskitorque2_args.argptr[:]
	cu.LaunchKernel(addslonczewskitorque2_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, stream0, args)

	if Synchronous { // debug
		Sync()
		timer.Stop("addslonczewskitorque2")
	}
}

// maps compute capability on PTX code for addslonczewskitorque2 kernel.
var addslonczewskitorque2_map = map[int]string{0: "",
	70: addslonczewskitorque2_ptx_70}

// addslonczewskitorque2 PTX code for various compute capabilities.
const (
	addslonczewskitorque2_ptx_70 = `
.version 7.2
.target sm_70
.address_size 64

	// .globl	addslonczewskitorque2

.visible .entry addslonczewskitorque2(
	.param .u64 addslonczewskitorque2_param_0,
	.param .u64 addslonczewskitorque2_param_1,
	.param .u64 addslonczewskitorque2_param_2,
	.param .u64 addslonczewskitorque2_param_3,
	.param .u64 addslonczewskitorque2_param_4,
	.param .u64 addslonczewskitorque2_param_5,
	.param .u64 addslonczewskitorque2_param_6,
	.param .f32 addslonczewskitorque2_param_7,
	.param .u64 addslonczewskitorque2_param_8,
	.param .f32 addslonczewskitorque2_param_9,
	.param .u64 addslonczewskitorque2_param_10,
	.param .f32 addslonczewskitorque2_param_11,
	.param .u64 addslonczewskitorque2_param_12,
	.param .f32 addslonczewskitorque2_param_13,
	.param .u64 addslonczewskitorque2_param_14,
	.param .f32 addslonczewskitorque2_param_15,
	.param .u64 addslonczewskitorque2_param_16,
	.param .f32 addslonczewskitorque2_param_17,
	.param .u64 addslonczewskitorque2_param_18,
	.param .f32 addslonczewskitorque2_param_19,
	.param .u64 addslonczewskitorque2_param_20,
	.param .f32 addslonczewskitorque2_param_21,
	.param .u64 addslonczewskitorque2_param_22,
	.param .f32 addslonczewskitorque2_param_23,
	.param .u64 addslonczewskitorque2_param_24,
	.param .f32 addslonczewskitorque2_param_25,
	.param .f32 addslonczewskitorque2_param_26,
	.param .f32 addslonczewskitorque2_param_27,
	.param .u32 addslonczewskitorque2_param_28
)
{
	.reg .pred 	%p<17>;
	.reg .f32 	%f<120>;
	.reg .b32 	%r<86>;
	.reg .f64 	%fd<3>;
	.reg .b64 	%rd<61>;


	ld.param.u64 	%rd1, [addslonczewskitorque2_param_0];
	ld.param.u64 	%rd2, [addslonczewskitorque2_param_1];
	ld.param.u64 	%rd3, [addslonczewskitorque2_param_2];
	ld.param.u64 	%rd4, [addslonczewskitorque2_param_3];
	ld.param.u64 	%rd5, [addslonczewskitorque2_param_4];
	ld.param.u64 	%rd6, [addslonczewskitorque2_param_5];
	ld.param.u64 	%rd7, [addslonczewskitorque2_param_6];
	ld.param.f32 	%f114, [addslonczewskitorque2_param_7];
	ld.param.u64 	%rd8, [addslonczewskitorque2_param_8];
	ld.param.f32 	%f109, [addslonczewskitorque2_param_9];
	ld.param.u64 	%rd9, [addslonczewskitorque2_param_10];
	ld.param.f32 	%f110, [addslonczewskitorque2_param_11];
	ld.param.u64 	%rd10, [addslonczewskitorque2_param_12];
	ld.param.f32 	%f111, [addslonczewskitorque2_param_13];
	ld.param.u64 	%rd11, [addslonczewskitorque2_param_14];
	ld.param.f32 	%f112, [addslonczewskitorque2_param_15];
	ld.param.u64 	%rd12, [addslonczewskitorque2_param_16];
	ld.param.f32 	%f115, [addslonczewskitorque2_param_17];
	ld.param.u64 	%rd13, [addslonczewskitorque2_param_18];
	ld.param.f32 	%f116, [addslonczewskitorque2_param_19];
	ld.param.u64 	%rd14, [addslonczewskitorque2_param_20];
	ld.param.f32 	%f117, [addslonczewskitorque2_param_21];
	ld.param.u64 	%rd15, [addslonczewskitorque2_param_22];
	ld.param.f32 	%f118, [addslonczewskitorque2_param_23];
	ld.param.u64 	%rd16, [addslonczewskitorque2_param_24];
	ld.param.f32 	%f119, [addslonczewskitorque2_param_25];
	ld.param.f32 	%f40, [addslonczewskitorque2_param_26];
	ld.param.f32 	%f41, [addslonczewskitorque2_param_27];
	ld.param.u32 	%r2, [addslonczewskitorque2_param_28];
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r4, %r3, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	setp.ge.s32 	%p1, %r1, %r2;
	@%p1 bra 	LBB0_25;

	cvta.to.global.u64 	%rd17, %rd4;
	mul.wide.s32 	%rd18, %r1, 4;
	add.s64 	%rd19, %rd17, %rd18;
	ld.global.nc.f32 	%f1, [%rd19];
	cvta.to.global.u64 	%rd20, %rd5;
	add.s64 	%rd21, %rd20, %rd18;
	ld.global.nc.f32 	%f2, [%rd21];
	cvta.to.global.u64 	%rd22, %rd6;
	add.s64 	%rd23, %rd22, %rd18;
	ld.global.nc.f32 	%f3, [%rd23];
	setp.eq.s64 	%p2, %rd8, 0;
	@%p2 bra 	LBB0_3;

	cvta.to.global.u64 	%rd24, %rd8;
	add.s64 	%rd26, %rd24, %rd18;
	ld.global.nc.f32 	%f42, [%rd26];
	mul.f32 	%f109, %f42, %f109;

LBB0_3:
	setp.eq.s64 	%p3, %rd9, 0;
	@%p3 bra 	LBB0_5;

	cvta.to.global.u64 	%rd27, %rd9;
	add.s64 	%rd29, %rd27, %rd18;
	ld.global.nc.f32 	%f43, [%rd29];
	mul.f32 	%f110, %f43, %f110;

LBB0_5:
	setp.eq.s64 	%p4, %rd10, 0;
	@%p4 bra 	LBB0_7;

	cvta.to.global.u64 	%rd30, %rd10;
	add.s64 	%rd32, %rd30, %rd18;
	ld.global.nc.f32 	%f44, [%rd32];
	mul.f32 	%f111, %f44, %f111;

LBB0_7:
	setp.eq.s64 	%p5, %rd11, 0;
	@%p5 bra 	LBB0_9;

	cvta.to.global.u64 	%rd33, %rd11;
	add.s64 	%rd35, %rd33, %rd18;
	ld.global.nc.f32 	%f45, [%rd35];
	mul.f32 	%f112, %f45, %f112;

LBB0_9:
	mul.f32 	%f47, %f111, %f111;
	fma.rn.f32 	%f48, %f110, %f110, %f47;
	fma.rn.f32 	%f49, %f112, %f112, %f48;
	sqrt.rn.f32 	%f12, %f49;
	setp.eq.f32 	%p6, %f12, 0f00000000;
	mov.f32 	%f113, 0f00000000;
	@%p6 bra 	LBB0_11;

	rcp.rn.f32 	%f113, %f12;

LBB0_11:
	mul.f32 	%f15, %f110, %f113;
	mul.f32 	%f16, %f111, %f113;
	mul.f32 	%f17, %f112, %f113;
	setp.eq.s64 	%p7, %rd7, 0;
	@%p7 bra 	LBB0_13;

	cvta.to.global.u64 	%rd36, %rd7;
	add.s64 	%rd38, %rd36, %rd18;
	ld.global.nc.f32 	%f50, [%rd38];
	mul.f32 	%f114, %f50, %f114;

LBB0_13:
	setp.eq.s64 	%p8, %rd12, 0;
	@%p8 bra 	LBB0_15;

	cvta.to.global.u64 	%rd39, %rd12;
	add.s64 	%rd41, %rd39, %rd18;
	ld.global.nc.f32 	%f51, [%rd41];
	mul.f32 	%f115, %f51, %f115;

LBB0_15:
	setp.eq.s64 	%p9, %rd13, 0;
	@%p9 bra 	LBB0_17;

	cvta.to.global.u64 	%rd42, %rd13;
	add.s64 	%rd44, %rd42, %rd18;
	ld.global.nc.f32 	%f52, [%rd44];
	mul.f32 	%f116, %f52, %f116;

LBB0_17:
	setp.eq.s64 	%p10, %rd14, 0;
	@%p10 bra 	LBB0_19;

	cvta.to.global.u64 	%rd45, %rd14;
	add.s64 	%rd47, %rd45, %rd18;
	ld.global.nc.f32 	%f53, [%rd47];
	mul.f32 	%f117, %f53, %f117;

LBB0_19:
	setp.eq.s64 	%p11, %rd15, 0;
	@%p11 bra 	LBB0_21;

	cvta.to.global.u64 	%rd48, %rd15;
	add.s64 	%rd50, %rd48, %rd18;
	ld.global.nc.f32 	%f54, [%rd50];
	mul.f32 	%f118, %f54, %f118;

LBB0_21:
	setp.eq.s64 	%p12, %rd16, 0;
	@%p12 bra 	LBB0_23;

	cvta.to.global.u64 	%rd51, %rd16;
	add.s64 	%rd53, %rd51, %rd18;
	ld.global.nc.f32 	%f55, [%rd53];
	mul.f32 	%f119, %f55, %f119;

LBB0_23:
	setp.eq.f32 	%p13, %f114, 0f00000000;
	setp.eq.f32 	%p14, %f109, 0f00000000;
	or.pred  	%p15, %p14, %p13;
	@%p15 bra 	LBB0_25;

	setp.eq.f32 	%p16, %f119, 0f00000000;
	selp.f32 	%f56, %f40, %f119, %p16;
	mul.f32 	%f57, %f56, %f41;
	mul.f32 	%f58, %f114, %f57;
	div.rn.f32 	%f59, %f109, %f58;
	cvt.f64.f32 	%fd1, %f59;
	mul.f64 	%fd2, %fd1, 0d3CC7B6EF14E9250C;
	cvt.rn.f32.f64 	%f60, %fd2;
	mul.f32 	%f61, %f117, %f117;
	mul.f32 	%f62, %f116, %f61;
	add.f32 	%f63, %f61, 0f3F800000;
	add.f32 	%f64, %f61, 0fBF800000;
	mul.f32 	%f65, %f2, %f16;
	fma.rn.f32 	%f66, %f1, %f15, %f65;
	fma.rn.f32 	%f67, %f3, %f17, %f66;
	fma.rn.f32 	%f68, %f67, %f64, %f63;
	div.rn.f32 	%f69, %f62, %f68;
	mul.f32 	%f70, %f69, %f60;
	mul.f32 	%f71, %f118, %f60;
	fma.rn.f32 	%f72, %f115, %f115, 0f3F800000;
	rcp.rn.f32 	%f73, %f72;
	fma.rn.f32 	%f74, %f115, %f71, %f70;
	mul.f32 	%f75, %f73, %f74;
	mul.f32 	%f76, %f115, %f70;
	sub.f32 	%f77, %f71, %f76;
	mul.f32 	%f78, %f73, %f77;
	mul.f32 	%f79, %f2, %f17;
	mul.f32 	%f80, %f3, %f16;
	sub.f32 	%f81, %f80, %f79;
	mul.f32 	%f82, %f3, %f15;
	mul.f32 	%f83, %f1, %f17;
	sub.f32 	%f84, %f83, %f82;
	mul.f32 	%f85, %f1, %f16;
	mul.f32 	%f86, %f2, %f15;
	sub.f32 	%f87, %f86, %f85;
	mul.f32 	%f88, %f2, %f87;
	mul.f32 	%f89, %f3, %f84;
	sub.f32 	%f90, %f88, %f89;
	mul.f32 	%f91, %f3, %f81;
	mul.f32 	%f92, %f1, %f87;
	sub.f32 	%f93, %f91, %f92;
	mul.f32 	%f94, %f1, %f84;
	mul.f32 	%f95, %f2, %f81;
	sub.f32 	%f96, %f94, %f95;
	mul.f32 	%f97, %f81, %f78;
	fma.rn.f32 	%f98, %f90, %f75, %f97;
	cvta.to.global.u64 	%rd54, %rd1;
	add.s64 	%rd56, %rd54, %rd18;
	ld.global.f32 	%f99, [%rd56];
	add.f32 	%f100, %f98, %f99;
	st.global.f32 	[%rd56], %f100;
	mul.f32 	%f101, %f84, %f78;
	fma.rn.f32 	%f102, %f93, %f75, %f101;
	cvta.to.global.u64 	%rd57, %rd2;
	add.s64 	%rd58, %rd57, %rd18;
	ld.global.f32 	%f103, [%rd58];
	add.f32 	%f104, %f102, %f103;
	st.global.f32 	[%rd58], %f104;
	mul.f32 	%f105, %f87, %f78;
	fma.rn.f32 	%f106, %f96, %f75, %f105;
	cvta.to.global.u64 	%rd59, %rd3;
	add.s64 	%rd60, %rd59, %rd18;
	ld.global.f32 	%f107, [%rd60];
	add.f32 	%f108, %f106, %f107;
	st.global.f32 	[%rd60], %f108;

LBB0_25:
	ret;

}

`
)
