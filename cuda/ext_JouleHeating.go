package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// Thermal equation for 1T model

func Evaldt0(temp0, dt0, m *data.Slice, Kth SymmLUT, Cth, Dth, Tsubsth, Tausubsth, res, Qext, J MSlice, mesh *data.Mesh, vol *data.Slice, regions *Bytes) {
	c := mesh.CellSize()
	N := mesh.Size()
	//	pbc := mesh.PBC_code()
	cfg := make3DConf(N)

	NN := mesh.Size()

	k_evaldt0_async(
		temp0.DevPtr(0), dt0.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		//Kth.DevPtr(0), Kth.Mul(0),
		unsafe.Pointer(Kth),
		Cth.DevPtr(0), Cth.Mul(0),
		Dth.DevPtr(0), Dth.Mul(0),
		Tsubsth.DevPtr(0), Tsubsth.Mul(0),
		Tausubsth.DevPtr(0), Tausubsth.Mul(0),
		res.DevPtr(0), res.Mul(0),
		Qext.DevPtr(0), Qext.Mul(0),
		J.DevPtr(X), J.Mul(X),
		J.DevPtr(Y), J.Mul(Y),
		J.DevPtr(Z), J.Mul(Z),
		float32(c[X]), float32(c[Y]), float32(c[Z]), NN[X], NN[Y], NN[Z],
		vol.DevPtr(0),
		regions.Ptr,
		cfg)
}

// Thermal equation for 2T model

func Evaldt02T(temp0e, dt0e, temp0l, dt0l, m *data.Slice, Ke SymmLUT, Ce MSlice, Kl SymmLUT, Cl, Gel, Dth, Tsubsth, Tausubsth, res, Qext, CD, J MSlice, mesh *data.Mesh, vol *data.Slice, regions *Bytes) {
	c := mesh.CellSize()
	N := mesh.Size()
	//	pbc := mesh.PBC_code()
	cfg := make3DConf(N)

	NN := mesh.Size()

	k_evaldt02T_async(
		temp0e.DevPtr(0), dt0e.DevPtr(0),
		temp0l.DevPtr(0), dt0l.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		//		Ke.DevPtr(0), Ke.Mul(0),
		unsafe.Pointer(Ke),
		Ce.DevPtr(0), Ce.Mul(0),
		//		Kl.DevPtr(0), Kl.Mul(0),
		unsafe.Pointer(Kl),
		Cl.DevPtr(0), Cl.Mul(0),
		Gel.DevPtr(0), Gel.Mul(0),
		Dth.DevPtr(0), Dth.Mul(0),
		Tsubsth.DevPtr(0), Tsubsth.Mul(0),
		Tausubsth.DevPtr(0), Tausubsth.Mul(0),
		res.DevPtr(0), res.Mul(0),
		Qext.DevPtr(0), Qext.Mul(0),
		CD.DevPtr(X), CD.Mul(X),
		CD.DevPtr(Y), CD.Mul(Y),
		CD.DevPtr(Z), CD.Mul(Z),
		J.DevPtr(X), J.Mul(X),
		J.DevPtr(Y), J.Mul(Y),
		J.DevPtr(Z), J.Mul(Z),
		float32(c[X]), float32(c[Y]), float32(c[Z]), NN[X], NN[Y], NN[Z],
		vol.DevPtr(0),
		regions.Ptr,
		cfg)
}

// Thermal energy 2T

func Add2TThermalEnergyDensity(dst, Te, Tl *data.Slice, Ce, Cl MSlice) {
	N := dst.Len()
	cfg := make1DConf(N)
	k_addthermalenergydensity_async(dst.DevPtr(X),
		Te.DevPtr(0), Tl.DevPtr(0), Ce.DevPtr(0), Ce.Mul(0), Cl.DevPtr(0), Cl.Mul(0), N, cfg)
}
