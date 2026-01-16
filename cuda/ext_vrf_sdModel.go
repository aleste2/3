package cuda

import (
	//	"unsafe"
	"github.com/mumax/3/data"
)

// Difussion equatiom for spin difussion
func MusStep(dmus *data.Slice, dt0 float32, m, mus *data.Slice, Rhosd, Tausf, dbar, sigmabar MSlice, dy11, dy1 *data.Slice, mesh *data.Mesh, vol *data.Slice, regions *Bytes) {
	c := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	NN := mesh.Size()

	k_evalmusStep_async(
		dmus.DevPtr(X), dmus.DevPtr(Y), dmus.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		mus.DevPtr(X), mus.DevPtr(Y), mus.DevPtr(Z),
		Rhosd.DevPtr(0), Rhosd.Mul(0),
		Tausf.DevPtr(0), Tausf.Mul(0),
		dbar.DevPtr(0), dbar.Mul(0),
		sigmabar.DevPtr(0), sigmabar.Mul(0),
		dy11.DevPtr(X), dy11.DevPtr(Y), dy11.DevPtr(Z),
		dy1.DevPtr(X), dy1.DevPtr(Y), dy1.DevPtr(Z),
		dt0,
		float32(c[X]), float32(c[Y]), float32(c[Z]), NN[X], NN[Y], NN[Z],
		vol.DevPtr(0),
		regions.Ptr,
		cfg)
}

func TTMplus(dy11, dy1, Te, m *data.Slice, Tc, rho, Ce MSlice, dt float32) {

	N := Te.Len()

	cfg := make1DConf(N)
	k_ttmplus_async(
		dy11.DevPtr(X), dy11.DevPtr(Y), dy11.DevPtr(Z),
		dy1.DevPtr(X), dy1.DevPtr(Y), dy1.DevPtr(Z),
		Te.DevPtr(X),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Tc.DevPtr(0), Tc.Mul(0),
		rho.DevPtr(0), rho.Mul(0),
		dt,
		Ce.DevPtr(0), Ce.Mul(0),
		N, cfg)
}

func AddMagneticmoment(m, mus *data.Slice, Tc, Tausd MSlice, dt float32) {
	N := m.Len()

	cfg := make1DConf(N)
	k_addmagneticmoment_async(
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		mus.DevPtr(X), mus.DevPtr(Y), mus.DevPtr(Z),
		Tc.DevPtr(0), Tc.Mul(0),
		Tausd.DevPtr(0), Tausd.Mul(0),
		dt,
		N, cfg)
}

func MusEffectiveField(dst, dmus, m *data.Slice, Tc, Ms MSlice) {
	N := dmus.Len()
	cfg := make1DConf(N)

	k_museffectivefield_async(
		dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		dmus.DevPtr(X), dmus.DevPtr(Y), dmus.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Tc.DevPtr(0), Tc.Mul(0),
		Ms.DevPtr(0), Ms.Mul(0),
		N, cfg)
}

func M3TMtorque(dst1, m, mus, Te *data.Slice, Tc, tausd MSlice) {
	N := m.Len()
	cfg := make1DConf(N)
	k_evalM3TMtorque_async(
		dst1.DevPtr(X), dst1.DevPtr(Y), dst1.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		mus.DevPtr(X), mus.DevPtr(Y), mus.DevPtr(Z),
		Te.DevPtr(X),
		Tc.DevPtr(0), Tc.Mul(0),
		tausd.DevPtr(0), tausd.Mul(0),
		N, cfg)
}

func AddJspin(dst1, mus *data.Slice, sbar MSlice, mesh *data.Mesh, vol *data.Slice) {
	c := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	NN := mesh.Size()
	k_addJspin_async(
		dst1.DevPtr(X), dst1.DevPtr(Y), dst1.DevPtr(Z),
		mus.DevPtr(X), mus.DevPtr(Y), mus.DevPtr(Z),
		sbar.DevPtr(0), sbar.Mul(0),
		float32(c[X]), float32(c[Y]), float32(c[Z]), NN[X], NN[Y], NN[Z],
		vol.DevPtr(0),
		cfg)
}
