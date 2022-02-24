package cuda

import (
	"github.com/mumax/3/data"
	//	"github.com/mumax/3/util"
	"unsafe"
)

func NormalizeAF(m0, m1, m2 *data.Slice, Ms0, Ms1, Ms2 MSlice) {
	N := m0.Len()
	cfg := make1DConf(N)
	k_normalizeAF_async(m0.DevPtr(X), m0.DevPtr(Y), m0.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		Ms0.DevPtr(0), Ms0.Mul(0),
		Ms1.DevPtr(0), Ms1.Mul(0),
		Ms2.DevPtr(0), Ms2.Mul(0), N, cfg)
}

func AddExchangeAFCell(dst1, dst2, m1, m2 *data.Slice, Ms1, Ms2, bex12, bex21 MSlice) {
	N := m1.Len()
	cfg := make1DConf(N)
	k_addExchangeAFCell_async(dst1.DevPtr(X), dst1.DevPtr(Y), dst1.DevPtr(Z),
		dst2.DevPtr(X), dst2.DevPtr(Y), dst2.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		Ms1.DevPtr(0), Ms1.Mul(0),
		Ms2.DevPtr(0), Ms2.Mul(0),
		bex12.DevPtr(0), bex12.Mul(0),
		bex21.DevPtr(0), bex21.Mul(0),
		N, cfg)
}

func AddExchangeAFll(dst1, dst2, m1, m2 *data.Slice, ms1, ms2 MSlice, llex_red SymmLUT, regions *Bytes, mesh *data.Mesh) {
	//func AddExchange(B, m *data.Slice, Aex_red SymmLUT, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(1 / (c[X] * c[X]))
	wy := float32(1 / (c[Y] * c[Y]))
	wz := float32(1 / (c[Z] * c[Z]))
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)

	k_addexchange_async(dst1.DevPtr(X), dst1.DevPtr(Y), dst1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		ms1.DevPtr(0), ms1.Mul(0),
		unsafe.Pointer(llex_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
	k_addexchange_async(dst2.DevPtr(X), dst2.DevPtr(Y), dst2.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		ms2.DevPtr(0), ms2.Mul(0),
		unsafe.Pointer(llex_red), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)

	/*
		k_addexchangeAfll_async(dst1.DevPtr(X), dst1.DevPtr(Y), dst1.DevPtr(Z),
			dst2.DevPtr(X), dst2.DevPtr(Y), dst2.DevPtr(Z),
			m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
			m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
			ms1.DevPtr(0), ms1.Mul(0),
			ms2.DevPtr(0), ms2.Mul(0),
			unsafe.Pointer(llex_red), regions.Ptr,
			wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)*/
}
