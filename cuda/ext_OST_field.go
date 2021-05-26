package cuda

import (
	"github.com/mumax/3/data"
	//"github.com/mumax/3/util"
)

// Add OST field to Beff.
// see uniaxialanisotropy.cu
func AddOSTField(Beff, S *data.Slice, Msat, Jex MSlice) {
	//util.Argument(Beff.Size() == m.Size())

	N := Beff.Len()
	cfg := make1DConf(N)
	k_addOSTField_async(
		Beff.DevPtr(X), Beff.DevPtr(Y), Beff.DevPtr(Z),
		S.DevPtr(X),S.DevPtr(Y), S.DevPtr(Z),
		Msat.DevPtr(0), Msat.Mul(0),
		Jex.DevPtr(0), Jex.Mul(0),
		N, cfg)
}
