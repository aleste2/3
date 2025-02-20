package cuda

import (
	"github.com/mumax/3/data"
)

// Elastic calculations
func CalcDU(dst, sigma, u *data.Slice, eta, rho, force MSlice, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(1 / c[X])
	wy := float32(1 / c[Y])
	wz := float32(1 / c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_calc_DU_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		sigma.DevPtr(X), sigma.DevPtr(Y), sigma.DevPtr(Z),
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z),
		eta.DevPtr(0), eta.Mul(0),
		rho.DevPtr(0), rho.Mul(0),
		force.DevPtr(0), force.Mul(0),
		wx, wy, wz, N[X], N[Y], N[Z], cfg)
}

func CalcDSigma(dst, u *data.Slice, c11, c12, c44 MSlice, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(1 / c[X])
	wy := float32(1 / c[Y])
	wz := float32(1 / c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_calc_Sigma_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z),
		c11.DevPtr(0), c11.Mul(0),
		c12.DevPtr(0), c12.Mul(0),
		c44.DevPtr(0), c44.Mul(0),
		wx, wy, wz, N[X], N[Y], N[Z], cfg)
}

// Magnetoelastic calculations
func CalcDSigmam(dst, u *data.Slice, c11, c12, c44, b1, b2 MSlice, mesh *data.Mesh, m, mold *data.Slice, deltat float32) {
	c := mesh.CellSize()
	wx := float32(1 / c[X])
	wy := float32(1 / c[Y])
	wz := float32(1 / c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_calc_Sigmam_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		u.DevPtr(X), u.DevPtr(Y), u.DevPtr(Z),
		c11.DevPtr(0), c11.Mul(0),
		c12.DevPtr(0), c12.Mul(0),
		c44.DevPtr(0), c44.Mul(0),
		b1.DevPtr(0), b1.Mul(0),
		b2.DevPtr(0), b2.Mul(0),
		wx, wy, wz, N[X], N[Y], N[Z],
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		mold.DevPtr(X), mold.DevPtr(Y), mold.DevPtr(Z),
		deltat, cfg)
}

// ME field

func AddMEField(dst, M, r *data.Slice, c11, c12, c44, b1, b2, ms MSlice, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(1 / c[X])
	wy := float32(1 / c[Y])
	wz := float32(1 / c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_addMEField_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		M.DevPtr(X), M.DevPtr(Y), M.DevPtr(Z),
		r.DevPtr(X), r.DevPtr(Y), r.DevPtr(Z),
		c11.DevPtr(0), c11.Mul(0),
		c12.DevPtr(0), c12.Mul(0),
		c44.DevPtr(0), c44.Mul(0),
		b1.DevPtr(0), b1.Mul(0),
		b2.DevPtr(0), b2.Mul(0),
		ms.DevPtr(0), ms.Mul(0),
		wx, wy, wz, N[X], N[Y], N[Z], cfg)
}

func AddMEField2(dst, M, sigma *data.Slice, c11, c12, c44, b1, b2, ms MSlice, mesh *data.Mesh) {
	c := mesh.CellSize()
	vol := float32(c[X] * c[Y] * c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_addMEField2_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		M.DevPtr(X), M.DevPtr(Y), M.DevPtr(Z),
		sigma.DevPtr(X), sigma.DevPtr(Y), sigma.DevPtr(Z),
		c11.DevPtr(0), c11.Mul(0),
		c12.DevPtr(0), c12.Mul(0),
		c44.DevPtr(0), c44.Mul(0),
		b1.DevPtr(0), b1.Mul(0),
		b2.DevPtr(0), b2.Mul(0),
		ms.DevPtr(0), ms.Mul(0),
		vol, N[X], N[Y], N[Z], cfg)
}

func AddStrain(dst, r *data.Slice, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(1 / c[X])
	wy := float32(1 / c[Y])
	wz := float32(1 / c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_addStrain_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		r.DevPtr(X), r.DevPtr(Y), r.DevPtr(Z),
		wx, wy, wz, N[X], N[Y], N[Z], cfg)
}
