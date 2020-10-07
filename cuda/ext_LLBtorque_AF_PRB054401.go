package cuda

import (
	"github.com/mumax/3/data"
)

// Landau-Lifshitz torque for AF implementation PRB 100 054401 (2019)
func LLBTorqueAFPRB(torque1, m1, torque2, m2, B1, B2 *data.Slice, temp MSlice, alpha, alpha1, alpha2, TCurie, Msat, Msat1, Msat2 MSlice, hth1a *data.Slice, hth2a *data.Slice, hth1b *data.Slice, hth2b *data.Slice, x, nv, mua, mub, J0aa, J0bb, J0ab, lambda0 MSlice) {
	N := torque1.Len()
	cfg := make1DConf(N)
	k_LLBtorqueAFPRB054401_async(torque1.DevPtr(X), torque1.DevPtr(Y), torque1.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		torque2.DevPtr(X), torque2.DevPtr(Y), torque2.DevPtr(Z),
		m2.DevPtr(X), m2.DevPtr(Y), m2.DevPtr(Z),
		B1.DevPtr(X), B1.DevPtr(Y), B1.DevPtr(Z),
		B2.DevPtr(X), B2.DevPtr(Y), B2.DevPtr(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		alpha1.DevPtr(0), alpha1.Mul(0),
		alpha2.DevPtr(0), alpha2.Mul(0),
		TCurie.DevPtr(0), TCurie.Mul(0),
		Msat.DevPtr(0), Msat.Mul(0),
		Msat1.DevPtr(0), Msat1.Mul(0),
		Msat2.DevPtr(0), Msat2.Mul(0),
		hth1a.DevPtr(X), hth1a.DevPtr(Y), hth1a.DevPtr(Z),
		hth2a.DevPtr(X), hth2a.DevPtr(Y), hth2a.DevPtr(Z),
		hth1b.DevPtr(X), hth1b.DevPtr(Y), hth1b.DevPtr(Z),
		hth2b.DevPtr(X), hth2b.DevPtr(Y), hth2b.DevPtr(Z),
		temp.DevPtr(0), temp.Mul(0),
		x.DevPtr(0), x.Mul(0),
		nv.DevPtr(0), nv.Mul(0),
		mua.DevPtr(0), mua.Mul(0),
		mub.DevPtr(0), mub.Mul(0),
		J0aa.DevPtr(0), J0aa.Mul(0),
		J0bb.DevPtr(0), J0bb.Mul(0),
		J0ab.DevPtr(0), J0ab.Mul(0),
		lambda0.DevPtr(0), lambda0.Mul(0),
		N, cfg)
}
