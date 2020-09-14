package cuda

import (
	"github.com/mumax/3/data"
)

// Landau-Lifshitz torque for AF implementation PRB 100 054401 (2019)
func LLBRenormAF2T(m01, m02, m1, m2 *data.Slice, temp *data.Slice, alpha, alpha1, alpha2, TCurie, Msat, Msat1, Msat2 MSlice, x, nv, mua, mub, J0aa, J0bb, J0ab MSlice, dt, Gamma1, Gamma2 float32) {
	N := m1.Len()
	cfg := make1DConf(N)
	k_AF2TRenorm_async(m01.DevPtr(X), m01.DevPtr(Y), m01.DevPtr(Z),
		m02.DevPtr(X), m02.DevPtr(Y), m02.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		alpha1.DevPtr(0), alpha1.Mul(0),
		alpha2.DevPtr(0), alpha2.Mul(0),
		TCurie.DevPtr(0), TCurie.Mul(0),
		Msat.DevPtr(0), Msat.Mul(0),
		Msat1.DevPtr(0), Msat1.Mul(0),
		Msat2.DevPtr(0), Msat2.Mul(0),
		temp.DevPtr(0),
		x.DevPtr(0), x.Mul(0),
		nv.DevPtr(0), nv.Mul(0),
		mua.DevPtr(0), mua.Mul(0),
		mub.DevPtr(0), mub.Mul(0),
		J0aa.DevPtr(0), J0aa.Mul(0),
		J0bb.DevPtr(0), J0bb.Mul(0),
		J0ab.DevPtr(0), J0ab.Mul(0),
		dt, Gamma1, Gamma2,
		N, cfg)
}

// Landau-Lifshitz torque for AF implementation PRB 100 054401 (2019)
func LLBRenormAF(m01, m02, m1, m2 *data.Slice, Temp, alpha, alpha1, alpha2, TCurie, Msat, Msat1, Msat2 MSlice, x, nv, mua, mub, J0aa, J0bb, J0ab MSlice, dt, Gamma1, Gamma2 float32) {
	N := m1.Len()
	cfg := make1DConf(N)
	k_AFRenorm_async(m01.DevPtr(X), m01.DevPtr(Y), m01.DevPtr(Z),
		m02.DevPtr(X), m02.DevPtr(Y), m02.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		alpha1.DevPtr(0), alpha1.Mul(0),
		alpha2.DevPtr(0), alpha2.Mul(0),
		TCurie.DevPtr(0), TCurie.Mul(0),
		Msat.DevPtr(0), Msat.Mul(0),
		Msat1.DevPtr(0), Msat1.Mul(0),
		Msat2.DevPtr(0), Msat2.Mul(0),
		Temp.DevPtr(0), Temp.Mul(0),
		x.DevPtr(0), x.Mul(0),
		nv.DevPtr(0), nv.Mul(0),
		mua.DevPtr(0), mua.Mul(0),
		mub.DevPtr(0), mub.Mul(0),
		J0aa.DevPtr(0), J0aa.Mul(0),
		J0bb.DevPtr(0), J0bb.Mul(0),
		J0ab.DevPtr(0), J0ab.Mul(0),
		dt, Gamma1, Gamma2,
		N, cfg)
}
