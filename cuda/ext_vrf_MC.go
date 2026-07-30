package cuda

import (
	"github.com/mumax/3/data"
	"unsafe"
)

// Change M
func MCmove(mnew, x, y, z *data.Slice, i int, delta float32, mesh *data.Mesh) {
	//c := mesh.CellSize()
	//wx := float32(1 / c[X])
	//wy := float32(1 / c[Y])
	//wz := float32(1 / c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_mcmove_async(mnew.DevPtr(X), mnew.DevPtr(Y), mnew.DevPtr(Z),
		x.DevPtr(X), y.DevPtr(X), z.DevPtr(X),
		i, delta,
		N[X], N[Y], N[Z], cfg)
}

func MCCompare(unew, uold, mnew, mold, rnd *data.Slice, temp MSlice, i int, mesh *data.Mesh, changed *data.Slice, cells float32) {
	c := mesh.CellSize()
	wx := float32(c[X])
	wy := float32(c[Y])
	wz := float32(c[Z])
	N := mesh.Size()
	cfg := make3DConf(N)
	k_mccompare_async(
		uold.DevPtr(X), unew.DevPtr(X),
		mold.DevPtr(X), mold.DevPtr(Y), mold.DevPtr(Z),
		mnew.DevPtr(X), mnew.DevPtr(Y), mnew.DevPtr(Z),
		rnd.DevPtr(X),
		temp.DevPtr(0), temp.Mul(0),
		changed.DevPtr(X),
		i,
		N[X], N[Y], N[Z], wx, wy, wz,
		cells, cfg)
}

func AddUexchangePlus(U, m, mold *data.Slice, Aex SymmLUT, Msat MSlice, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(2 / (c[X] * c[X]))
	wy := float32(2 / (c[Y] * c[Y]))
	wz := float32(2 / (c[Z] * c[Z]))
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_addUexchangePlus_async(U.DevPtr(X),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		mold.DevPtr(X), mold.DevPtr(Y), mold.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		unsafe.Pointer(Aex), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
}
