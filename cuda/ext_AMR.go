package cuda

import (
	"github.com/mumax/3/data"
)

// Step for AMR
func EvolveAMR(x1, x2 int, LocalV, LocalSigma, m, J_AMR *data.Slice,
	DeltaRho MSlice, mesh *data.Mesh) {

	c := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	NN := mesh.Size()

	k_evolveAMR_async(
		x1, x2,
		LocalV.DevPtr(0),
		LocalSigma.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		J_AMR.DevPtr(X), J_AMR.DevPtr(Y), J_AMR.DevPtr(Z),
		DeltaRho.DevPtr(0), DeltaRho.Mul(0),
		float32(c[X]), float32(c[Y]), float32(c[Z]), NN[X], NN[Y], NN[Z],
		cfg)
}

// J_AMR calculation
func CalculateJAMR(LocalV, LocalSigma, m, J_AMR *data.Slice,
	mesh *data.Mesh) {

	c := mesh.CellSize()
	N := mesh.Size()
	cfg := make3DConf(N)

	NN := mesh.Size()

	k_calculateJAMR_async(
		LocalV.DevPtr(0),
		LocalSigma.DevPtr(0),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		J_AMR.DevPtr(X), J_AMR.DevPtr(Y), J_AMR.DevPtr(Z),
		float32(c[X]), float32(c[Y]), float32(c[Z]), NN[X], NN[Y], NN[Z],
		cfg)
}
