package cuda

import (
	"github.com/mumax/3/data"
)

// Landau-Lifshitz torque for AF implementation PRB 100 054401 (2019)
func LLBTorqueMFA(torque1, m1, B1 *data.Slice, temp MSlice, alpha, Msat MSlice, hth1a *data.Slice, hth2a *data.Slice, nv, mua, J0aa MSlice) {
	N := torque1.Len()
	cfg := make1DConf(N)
	k_LLBtorqueMFA2_async(torque1.DevPtr(X), torque1.DevPtr(Y), torque1.DevPtr(Z),
		m1.DevPtr(X), m1.DevPtr(Y), m1.DevPtr(Z),
		B1.DevPtr(X), B1.DevPtr(Y), B1.DevPtr(Z),
		alpha.DevPtr(0), alpha.Mul(0),
		Msat.DevPtr(0), Msat.Mul(0),
		hth1a.DevPtr(X), hth1a.DevPtr(Y), hth1a.DevPtr(Z),
		hth2a.DevPtr(X), hth2a.DevPtr(Y), hth2a.DevPtr(Z),
		temp.DevPtr(0), temp.Mul(0),
		nv.DevPtr(0), nv.Mul(0),
		mua.DevPtr(0), mua.Mul(0),
		J0aa.DevPtr(0), J0aa.Mul(0),
		N, cfg)
}