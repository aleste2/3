package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"unsafe"
)

func Alloypar(host, alloy int, percent float64, random *data.Slice, regions unsafe.Pointer) {
	N := random.Len()
	cfg := make1DConf(N)
	k_alloyparcuda_async(byte(host), byte(alloy), float32(percent), random.DevPtr(0), unsafe.Pointer(regions),
		N, cfg)
}

// Set Bth to thermal noise (Brown).
// see temperature.cu
func SetTemperatureAto(Bth, noise *data.Slice, k2mu0_Mu0VgammaDt float64, Mu, Temp, Alpha MSlice, ScaleNoiseLLB float64) {
	util.Argument(Bth.NComp() == 1 && noise.NComp() == 1)

	N := Bth.Len()
	cfg := make1DConf(N)

	k_settemperature2_async(Bth.DevPtr(0), noise.DevPtr(0), float32(k2mu0_Mu0VgammaDt),
		Mu.DevPtr(0), Mu.Mul(0),
		Temp.DevPtr(0), Temp.Mul(0),
		Alpha.DevPtr(0), Alpha.Mul(0),
		float32(ScaleNoiseLLB),
		N, cfg)
}

// Add uniaxial magnetocrystalline anisotropy field to Beff.
// see uniaxialanisotropyato.cu
func AddUniaxialAnisotropyAto(Beff, m *data.Slice, Mu, Dato, u MSlice) {
	util.Argument(Beff.Size() == m.Size())

	checkSize(Beff, m, Dato, u, Mu)

	N := Beff.Len()
	cfg := make1DConf(N)

	k_adduniaxialanisotropyato_async(
		Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Mu.DevPtr(0), Mu.Mul(0),
		Dato.DevPtr(0), Dato.Mul(0),
		u.DevPtr(X), u.Mul(X),
		u.DevPtr(Y), u.Mul(Y),
		u.DevPtr(Z), u.Mul(Z),
		N, cfg)
}

// Add exchange field to Beff.
// see exchangeato.cu
func AddExchangeAto(B, m *data.Slice, Jato, Jdmi SymmLUT, Mu, Nv MSlice, regions *Bytes, mesh *data.Mesh) {
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_addexchangeato_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Mu.DevPtr(0), Mu.Mul(0),
		Nv.DevPtr(0), Nv.Mul(0),
		unsafe.Pointer(Jato), unsafe.Pointer(Jdmi), regions.Ptr,
		N[X], N[Y], N[Z], pbc, cfg)
}

// finally multiply effective field by lande factor
func MultiplyLandeFactor(B *data.Slice, lande MSlice) {
	N := B.Len()
	cfg := make1DConf(N)
	k_MultiplyLandeFactor_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		lande.DevPtr(0), lande.Mul(0),
		N, cfg)
}

func MultiplyVolume(B *data.Slice, mesh *data.Mesh, celltype int) {
	N := B.Len()
	c := mesh.CellSize()
	volume := float32(c[X] * c[Y] * c[Z])
	if celltype == 1 {
		volume = volume * 0.68 * 8
	}
	if celltype == 2 {
		volume = volume * 0.74 * 8
	}
	cfg := make1DConf(N)
	print(volume)
	k_MultiplyVolume_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		volume,
		N, cfg)
}

func AddSlonczewskiTorque2Ato(torque, m *data.Slice, Msat, J, fixedP, alpha, pol, λ, ε_prime MSlice, thickness MSlice, flp float64, mesh *data.Mesh, celltype int) {
	N := torque.Len()
	cfg := make1DConf(N)
	c := mesh.CellSize()
	meshThickness := mesh.WorldSize()[Z]
	volume := (c[X] * c[Y] * c[Z])
	if celltype == 1 {
		volume = volume * 0.68 * 8
	}
	if celltype == 2 {
		volume = volume * 0.74 * 8
	}
	flt := float32(flp * mesh.WorldSize()[Z] / volume) // Put factor for conversion from mu to Msat in flt
	//if (celltype>0) {flt=flt/2.0}

	k_addslonczewskitorque2_async(
		torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		J.DevPtr(Z), J.Mul(Z),
		fixedP.DevPtr(X), fixedP.Mul(X),
		fixedP.DevPtr(Y), fixedP.Mul(Y),
		fixedP.DevPtr(Z), fixedP.Mul(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		pol.DevPtr(0), pol.Mul(0),
		λ.DevPtr(0), λ.Mul(0),
		ε_prime.DevPtr(0), ε_prime.Mul(0),
		thickness.DevPtr(0), thickness.Mul(0),
		float32(meshThickness),
		float32(flt),
		N, cfg)
}
