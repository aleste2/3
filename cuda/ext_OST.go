package cuda

import (
	"github.com/mumax/3/data"
)

func StepOST(m, S, dy1 *data.Slice, tau, Jex, R, dir MSlice, mesh *data.Mesh) {
	c := mesh.CellSize()
	N := mesh.Size()
	//	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	NN := mesh.Size()

	k_evaldtOST_async(
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		S.DevPtr(X), S.DevPtr(Y), S.DevPtr(Z),
		dy1.DevPtr(X), dy1.DevPtr(Y), dy1.DevPtr(Z),
		tau.DevPtr(0), tau.Mul(0),
		Jex.DevPtr(0), Jex.Mul(0),
		R.DevPtr(0), R.Mul(0),
		dir.DevPtr(X), dir.Mul(X),
		dir.DevPtr(Y), dir.Mul(Y),
		dir.DevPtr(Z), dir.Mul(Z),
		float32(c[X]), float32(c[Y]), float32(c[Z]), NN[X], NN[Y], NN[Z],
		cfg)
}
