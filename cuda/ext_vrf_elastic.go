package cuda

import (
	"github.com/mumax/3/data"
)

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
