package cuda

import (
	"github.com/mumax/3/data"
	"unsafe"
)

// Evaluate chiral damping

func AlphaChiralEvaluate(dst, m *data.Slice, alphaC, Ku1 MSlice, Aex_red SymmLUT, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(c[X])
	wy := float32(c[Y])
	wz := float32(c[Z])
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_alphachiralLocal_async(dst.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		alphaC.DevPtr(0), alphaC.Mul(0),
		Ku1.DevPtr(0), Ku1.Mul(0),
		unsafe.Pointer(Aex_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
}
