package cuda

import (
	"github.com/mumax/3/data"
)

// Chiral Landau-Lifshitz torque

func ChiralLLTorque(torque, m, B *data.Slice, alpha MSlice, alphaC *data.Slice) {
	N := torque.Len()
	cfg := make1DConf(N)

	k_chirallltorque2_async(torque.DevPtr(X), torque.DevPtr(Y), torque.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		alpha.DevPtr(0), alpha.Mul(0), alphaC.DevPtr(X), N, cfg)
}
