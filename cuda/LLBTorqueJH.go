package cuda

import (
	"github.com/mumax/3/data"
)

// Landau-Lifshitz-Bloch torque divided by gamma0:
// 	- 1/(1+α²) [ m x B +  α m x (m x B) ]
//      m x B + αpar (m · B) m/(m·m) - αperp [m x (Beff + Hthp)]/(m·m) **** Hthpar will be added aferwards *****
// 	torque in Tesla
// 	m normalized
// 	B in Tesla
// see lltorque.cu
func LLBTorqueJH(torque, m, B *data.Slice, tempJH *data.Slice, alpha, TCurie, Msat MSlice, hth1 *data.Slice, hth2 *data.Slice, Langevin int, a1 MSlice) {
	N := torque.Len()
	cfg := make1DConf(N)
	if Langevin == 1 {
		k_LLBtorque4JH_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
			m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
			B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
			alpha.DevPtr(0), alpha.Mul(0),
			TCurie.DevPtr(0), TCurie.Mul(0),
			Msat.DevPtr(0), Msat.Mul(0),
			hth1.DevPtr(X), hth1.DevPtr(Y), hth1.DevPtr(Z),
			hth2.DevPtr(X), hth2.DevPtr(Y), hth2.DevPtr(Z),
			tempJH.DevPtr(0),
			a1.DevPtr(0), a1.Mul(0),
			N, cfg)
	} else {
		k_LLBtorque2JH_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
			m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
			B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
			alpha.DevPtr(0), alpha.Mul(0),
			TCurie.DevPtr(0), TCurie.Mul(0),
			Msat.DevPtr(0), Msat.Mul(0),
			hth1.DevPtr(X), hth1.DevPtr(Y), hth1.DevPtr(Z),
			hth2.DevPtr(X), hth2.DevPtr(Y), hth2.DevPtr(Z),
			tempJH.DevPtr(0),
			N, cfg)
	}
}
