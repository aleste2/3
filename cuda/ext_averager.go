package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Simple averager for vector field

func Averager(dst, m, mask *data.Slice, nv int) {
	N := m.Size()
	util.Argument(m.Size() == N)
	cfg := make3DConf(N)
	k_averager_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		mask.DevPtr(X),
		nv, N[X], N[Y], N[Z], cfg)
}
