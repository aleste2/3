package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
	"math"
)

// FCTS Joule Heating method, can be used as solver.Step
type OnlyJH struct {
}

func (SJH *OnlyJH) Step() {

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si)
	util.Assert(dt > 0)
	AdaptativeFTCSSstepJH(dt)
	Time += Dt_si
}

func (SJH *OnlyJH) Free() {
}

func AdaptativeFTCSSstepJH(dt float32) {

	te := Te.temp
	size := Te.temp.Size()

	y := M.Buffer()

	Cel := Ce.MSlice()
	defer Cel.Recycle()

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

	// Buffers
	DeltaTe := cuda.Buffer(1, size)
	defer cuda.Recycle(DeltaTe)

	nsteps := 1.0
	dtT := float64(dt)
	if (FixDtT != 0) && (float32(FixDtT) < dt) {
		nsteps = math.Ceil(float64(dt) / FixDtT)
		dtT = float64(float32(dt) / float32(nsteps))
	}

	// Estimate time step and addapt if needed
	Substeps := 1
	cuda.Evaldt0JH(te, DeltaTe, y, kel.Gpu(), Cel,
		res, Qext, CD, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
	toleranceT := 0.001
	err := cuda.MaxAbs(DeltaTe)
	if err*float32(dtT) > float32(toleranceT) {
		Substeps = int(math.Ceil(float64(err * float32(dtT) / float32(toleranceT))))
		if Substeps > 1000 {
			Substeps = 1000
		}
		nsteps = nsteps * float64(Substeps)
		dtT = dtT / float64(Substeps)
	}

	T0 := Time
	// Recalculate Te,Tl
	for i := 0; i < int(nsteps); i++ {
		cuda.Evaldt0JH(te, DeltaTe, y, kel.Gpu(), Cel,
			res, Qext, CD, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
		//err := cuda.MaxAbs(DeltaTe)
		cuda.Madd2(te, te, DeltaTe, 1, float32(dtT)) // temp = temp + dt * dtemp0
		Time = Time + dtT
	}

	// Tausubs if needed, oly for long times
	if Time < Lasttsubs {
		Lasttsubs = Time
	}
	cuda.Evaldt02Ttausubs(te, DeltaTe, Tsubsth, Tausubsth, M.Mesh(), geometry.Gpu())
	err = cuda.MaxAbs(DeltaTe)
	if float64(err)*(Time-Lasttsubs) > 0.00005 {
		//	if (float64(err)*(Time-Lasttsubs) > 0.00005) ||((Time - Lasttsubs)>1e-10 ){
		cuda.Madd2(te, te, DeltaTe, 1, float32((Time - Lasttsubs))) // temp = temp + dt * dtemp0
		Lasttsubs = Time
	}

	Time = T0

	NSteps++
}
