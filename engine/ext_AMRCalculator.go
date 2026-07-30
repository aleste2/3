package engine

import (
	"github.com/mumax/3/cuda"
	//"github.com/mumax/3/cuda/curand"
	//"github.com/mumax/3/data"
	//"github.com/mumax/3/mag"
	"math"
)

var (
	LocalV     LocalTemp     // Voltage temperature
	LocalSigma LocalTemp     // Local resistivity as AMR
	J_AMR      magnetization // Calculated current from voltage
	deltaRho   = NewScalarParam("deltaRho", "a.u.", "resistivity change")
)

func init() {
	LocalV.name = "Voltage"
	LocalSigma.name = "Conductivity"
	DeclFunc("ext_AMRCalculator", AMRCalculator, "AMR")
	DeclROnly("localV", AsScalarField(&LocalV), "AMR Voltage")
	DeclROnly("localSigma", AsScalarField(&LocalSigma), "AMR conductivity")
	DeclLValue("JAMR", &J_AMR, `JAMR`)
	J_AMR.name = "JAMR_"
}

func AMRCalculator(x1, x2 int) float64 {

	m := M.Buffer()
	J_AMR.alloc()
	LocalSigma.update()
	LocalV.update()
	Vant := LocalV.average()
	Vfin := LocalV.average()
	DeltaRho := deltaRho.MSlice()
	defer DeltaRho.Recycle()
	for {
		cuda.EvolveAMR(x1, x2, LocalV.temp, LocalSigma.temp, m, J_AMR.Buffer(), DeltaRho, M.Mesh())
		cuda.CalculateJAMR(LocalV.temp, LocalSigma.temp, m, J_AMR.Buffer(), M.Mesh())
		//	J_AMR.normalize()
		Vfin = LocalV.average()
		print(Vfin[0]-Vant[0], "\r")
		if (Vfin[0]-Vant[0])*(Vfin[0]-Vant[0]) == 0 {
			break
		}
		Vant = LocalV.average()
	}

	JAMR := J_AMR.Buffer()
	JAMR_x := JAMR.Comp(0).HostCopy().Scalars()
	JAMR_y := JAMR.Comp(1).HostCopy().Scalars()
	JAMR_z := JAMR.Comp(2).HostCopy().Scalars()
	acumX := float64(0)
	acumY := float64(0)
	acumZ := float64(0)
	s := J_AMR.Buffer().Size()
	Ny, Nz := s[Y], s[Z]

	// AMR Calculation from deltaV=0 and J (to integrate in pad1)
	for z := 0; z < Nz; z++ {
		for y := 0; y < Ny; y++ {
			acumX += float64(JAMR_x[z][y][0])
			acumY += float64(JAMR_y[z][y][0])
			acumZ += float64(JAMR_z[z][y][0])
		}
	}
	return (1.0 / math.Sqrt(acumX*acumX+acumY*acumY+acumZ*acumZ))
}
