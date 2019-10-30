package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
//	"github.com/mumax/3/data"
//		"math"
)


// Heun solver joule heating only equation.
type Only2T struct{}

// Adaptive HeunJH method, can be used as solver.Step
func (_ *Only2T) Step() {


	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si )
	util.Assert(dt > 0)

	// stage 1

    NewtonStep2T(dt)
    Time += Dt_si
}

func (_ *Only2T) Free() {}


func NewtonStep2T(dt float32) {
	te := Te.temp
	size := Te.temp.Size()
	tl := Tl.temp
	y := M.Buffer()

	Kel := Ke.MSlice()
	defer Kel.Recycle()
	Cel := Ce.MSlice()
	defer Cel.Recycle()

	Klat := Kl.MSlice()
	defer Klat.Recycle()
	Clat := Cl.MSlice()
	defer Clat.Recycle()

	Gellat := Gel.MSlice()
	defer Gellat.Recycle()

	Dth := Density.MSlice()
	defer Dth.Recycle()

	Tsubsth := TSubs.MSlice()
	defer Tsubsth.Recycle()
	Tausubsth := TauSubs.MSlice()
	defer Tausubsth.Recycle()
	res := Resistivity.MSlice()
	defer res.Recycle()
	Qext := Qext.MSlice()
	defer Qext.Recycle()
	j := J.MSlice()
	defer j.Recycle()
	
	CD := CD.MSlice()
	defer CD.Recycle()

	// backup temps
	DeltaTe := cuda.Buffer(1, size)
	defer cuda.Recycle(DeltaTe)
	DeltaTl := cuda.Buffer(1, size)
	defer cuda.Recycle(DeltaTl)

	cuda.Evaldt02T(te,DeltaTe,tl,DeltaTl,y,Kel,Cel,Klat,Clat,Gellat,Dth,Tsubsth,Tausubsth,res,Qext,CD,j,M.Mesh())
	err := cuda.MaxAbs(DeltaTe)
	if (err*dt<2e-4) {
		TOverstepsCounter++
		if (TOverstepsCounter>=100) {
			cuda.Madd2(te,te, DeltaTe, 1, dt*100) // temp = temp + dt * dtemp0
			cuda.Madd2(tl,tl, DeltaTl, 1, dt*100) // temp = temp + dt * dtemp0
			TOverstepsCounter=0
			//print("si\n")
		}
	} else {
	    cuda.Madd2(te,te, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
		cuda.Madd2(tl,tl, DeltaTl, 1, dt) // temp = temp + dt * dtemp0
		TOverstepsCounter=0
	}

	//setLastErr(float64(err))
	//Time += Dt_si

	//cuda.Evaldt02T(te,t0e,tl,t0l,y,Kel,Cel,Klat,Clat,Gellat,Dth,Tsubsth,Tausubsth,res,Qext,CD,j,M.Mesh())
	//cuda.Madd3(te, te, t0e, t0e, 1, 0.5*dt, -0.5*dt) //****
	//cuda.Madd3(tl, tl, t0l, t0l, 1, 0.5*dt, -0.5*dt) //****

}

