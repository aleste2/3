package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func SetTemperatureJH(Bth, noise *data.Slice, k2mu0_Mu0VgammaDt float64, Msat MSlice, TempJH *data.Slice, Alpha MSlice) {
	util.Argument(Bth.NComp() == 1 && noise.NComp() == 1)

	N := Bth.Len()
	cfg := make1DConf(N)

	k_settemperatureJH_async(Bth.DevPtr(0), noise.DevPtr(0), float32(k2mu0_Mu0VgammaDt),
		Msat.DevPtr(0), Msat.Mul(0),
		TempJH.DevPtr(0),
		Alpha.DevPtr(0), Alpha.Mul(0),
		N, cfg)
}

func InitTemperatureJH(TempJH *data.Slice, TSubs MSlice) {

	N := TempJH.Len()
	cfg := make1DConf(N)
	k_InittemperatureJH_async(TempJH.DevPtr(0),TSubs.DevPtr(0), TSubs.Mul(0),N, cfg)

}

func InitmLLB(m *data.Slice,temp,TCurie MSlice) {
	N := m.Len()
	cfg := make1DConf(N)
	k_initmLLB_async(m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),temp.DevPtr(0), temp.Mul(0),TCurie.DevPtr(0), TCurie.Mul(0),N, cfg)
}

func InitmLLBJH(m ,TempJH *data.Slice,TCurie MSlice) {
	N := m.Len()
	cfg := make1DConf(N)
	k_initmLLBJH_async(m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),TempJH.DevPtr(0),TCurie.DevPtr(0), TCurie.Mul(0),N, cfg)
}
