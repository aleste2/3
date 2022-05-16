package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
)

// Heun solver joule heating only equation.
type HeunJH struct{}

// Adaptive HeunJH method, can be used as solver.Step
func (_ *HeunJH) Step() {

	// Parameter for JH
	temp0 := TempJH.temp
	dtemp0 := cuda.Buffer(1, temp0.Size())
	defer cuda.Recycle(dtemp0)

	Kth := Kthermal.MSlice()
	defer Kth.Recycle()
	Cth := Cthermal.MSlice()
	defer Cth.Recycle()
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

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si * GammaLL)
	util.Assert(dt > 0)

	// stage 1

	// Rewrite to calculate m step 1
	cuda.Evaldt0(temp0, dtemp0, M.Buffer(), Kth, Cth, Dth, Tsubsth, Tausubsth, res, Qext, j, M.Mesh())
	cuda.Madd2(temp0, temp0, dtemp0, 1, dt/float32(GammaLL)) // temp = temp + dt * dtemp0

	// stage 2

	dtemp := cuda.Buffer(1, dtemp0.Size())
	defer cuda.Recycle(dtemp)
	Time += Dt_si

	// Rewrite to calculate spep 2
	cuda.Evaldt0(temp0, dtemp, M.Buffer(), Kth, Cth, Dth, Tsubsth, Tausubsth, res, Qext, j, M.Mesh())
	cuda.Madd3(temp0, temp0, dtemp, dtemp0, 1, 0.5*dt/float32(GammaLL), -0.5*dt/float32(GammaLL)) //****

}

func (_ *HeunJH) Free() {}

func StepJH(dt float32) {

	// Parameter for JH
	temp0 := Te.temp
	dtemp0 := cuda.Buffer(1, temp0.Size())
	defer cuda.Recycle(dtemp0)

	Kth := Kthermal.MSlice()
	defer Kth.Recycle()
	Cth := Cthermal.MSlice()
	defer Cth.Recycle()
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

	// stage 1

	// Rewrite to calculate m step 1
	cuda.Evaldt0(temp0, dtemp0, M.Buffer(), Kth, Cth, Dth, Tsubsth, Tausubsth, res, Qext, j, M.Mesh())
	cuda.Madd2(temp0, temp0, dtemp0, 1, dt) // temp = temp + dt * dtemp0

	// stage 2

	dtemp := cuda.Buffer(1, dtemp0.Size())
	defer cuda.Recycle(dtemp)
	//Time += Dt_si

	// Rewrite to calculate spep 2
	cuda.Evaldt0(temp0, dtemp, M.Buffer(), Kth, Cth, Dth, Tsubsth, Tausubsth, res, Qext, j, M.Mesh())
	cuda.Madd3(temp0, temp0, dtemp, dtemp0, 1, 0.5*dt, -0.5*dt) //****
}
