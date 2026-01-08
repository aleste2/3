package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Heun solver joule heating only equation.
type OnlyJH struct {
	bufferTe    *data.Slice // buffer for slow Te evolucion
	bufferTeBig *data.Slice // buffer for Te evolution
}

// Adaptive Newton method, can be used as solver.Step
func (SJH *OnlyJH) Step() {

	// first step ever: one-time buffer Tel init and eval
	if SJH.bufferTe == nil {
		size := Te.Mesh().Size()
		SJH.bufferTe = cuda.NewSlice(1, size)
		SJH.bufferTeBig = cuda.NewSlice(1, size)
		cuda.Madd2(SJH.bufferTeBig, Te.temp, Te.temp, 1, 0)
	}

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si)
	util.Assert(dt > 0)
	AdaptativeNewtonStepJH(dt, SJH.bufferTe, SJH.bufferTeBig)
	Time += Dt_si
}

func (SJH *OnlyJH) Free() {
	SJH.bufferTe.Free()
	SJH.bufferTe = nil
	SJH.bufferTeBig.Free()
	SJH.bufferTeBig = nil
}
/*
func AdaptativeNewtonStepJHold(dt float32, bufferTe, bufferTeBig *data.Slice) {

	te := Te.temp
	size := Te.temp.Size()

	// Parameter for JH
	//	Kth := Kthermal.MSlice()  // Redo with type exchange
	//	defer Kth.Recycle()
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

	// backup temps
	DeltaTe := cuda.Buffer(1, size)
	defer cuda.Recycle(DeltaTe)
	DeltaTl := cuda.Buffer(1, size)
	defer cuda.Recycle(DeltaTl)

    scaletausubs=0.0	
	// New scaling tausubs
	if (t-lasttsubs)>1e-13 {
		scaletausubs=(t-lasttsubs)/FixDT
	}


	// Recalculate Te,Tl
	UpdateTe(te, bufferTeBig, bufferTe)
	//	cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, kel.Gpu(), Cel, kll.Gpu(), Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
	cuda.Evaldt0(te, DeltaTe, M.Buffer(), kth.Gpu(), Cth, Dth, Tsubsth, Tausubsth,scaletausubs, res, Qext, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
	err := cuda.MaxAbs(DeltaTe)

	toleranceT := 1.0
	minStepTe := 0.1
	Substeps := 1

	if err*dt < float32(toleranceT) {
		if float64(err*dt) > minStepTe {
			cuda.Madd2(bufferTeBig, bufferTeBig, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
		} else {
			cuda.Madd2(bufferTe, bufferTe, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
			err2 := cuda.MaxAbs(bufferTe)
			// Add small buffers if needed
			if float64(err2) >= minStepTe {
				cuda.Madd2(bufferTeBig, bufferTeBig, bufferTe, 1, 1) // temp = temp + bufferTe
				cuda.Madd2(bufferTe, bufferTe, DeltaTe, 0, 0)        // bufferTe to 0
			}
		}
	} else {
		for {
			Substeps = Substeps * 2
			if (err * dt / float32(Substeps)) < float32(toleranceT) {
				break
			}
		}
		Substeps = int(math.Ceil(float64(float32(err*dt) / float32(toleranceT))))
		for iSubsteps := 0; iSubsteps < Substeps; iSubsteps++ {
			UpdateTe(te, bufferTeBig, bufferTe)
			//			cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, kel.Gpu(), Cel, kll.Gpu(), Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
			cuda.Evaldt0(te, DeltaTe, M.Buffer(), kth.Gpu(), Cth, Dth, Tsubsth, Tausubsth, res, Qext, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
			cuda.Madd2(bufferTeBig, bufferTeBig, DeltaTe, 1, dt/float32(Substeps)) // temp = temp + dt * dtemp0
			Time += float64(dt / float32(Substeps))
		}
		Time -= float64(dt)
	}

	
	UpdateTe(te, bufferTeBig, bufferTe)
}
*/
// Update Te with buffers
func UpdateTe(Te, bufferTeBig, bufferTe *data.Slice) {
	cuda.Madd2(Te, bufferTe, bufferTeBig, 1, 1)
}



func AdaptativeNewtonStepJH(dt float32, bufferTe, bufferTeBig *data.Slice) {

	te := Te.temp
	size := Te.temp.Size()

	// Parameter for JH
	//	Kth := Kthermal.MSlice()  // Redo with type exchange
	//	defer Kth.Recycle()
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

	// backup temps
	DeltaTe := cuda.Buffer(1, size)
	defer cuda.Recycle(DeltaTe)
	DeltaTl := cuda.Buffer(1, size)
	defer cuda.Recycle(DeltaTl)

	// Recalculate Te,Tl
	UpdateTe(te, bufferTeBig, bufferTe)
	//	cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, kel.Gpu(), Cel, kll.Gpu(), Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
	cuda.Evaldt0(te, DeltaTe, M.Buffer(), kth.Gpu(), Cth, Dth, Tsubsth, Tausubsth,
	 res, Qext, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
	err := cuda.MaxAbs(DeltaTe)

	toleranceT := 1.0
	minStepTe := 0.1
	Substeps := 1

	if err*dt < float32(toleranceT) {
		if float64(err*dt) > minStepTe {
			cuda.Madd2(bufferTeBig, bufferTeBig, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
		} else {
			cuda.Madd2(bufferTe, bufferTe, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
			err2 := cuda.MaxAbs(bufferTe)
			// Add small buffers if needed
			if float64(err2) >= minStepTe {
				cuda.Madd2(bufferTeBig, bufferTeBig, bufferTe, 1, 1) // temp = temp + bufferTe
				cuda.Madd2(bufferTe, bufferTe, DeltaTe, 0, 0)        // bufferTe to 0
			}
		}
	} else {
		for {
			Substeps = Substeps * 2
			if (err * dt / float32(Substeps)) < float32(toleranceT) {
				break
			}
		}
		Substeps = int(math.Ceil(float64(float32(err*dt) / float32(toleranceT))))
		for iSubsteps := 0; iSubsteps < Substeps; iSubsteps++ {
			UpdateTe(te, bufferTeBig, bufferTe)
			//			cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, kel.Gpu(), Cel, kll.Gpu(), Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
			cuda.Evaldt0(te, DeltaTe, M.Buffer(), kth.Gpu(), Cth, Dth, Tsubsth, Tausubsth, res, Qext, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
			cuda.Madd2(bufferTeBig, bufferTeBig, DeltaTe, 1, dt/float32(Substeps)) // temp = temp + dt * dtemp0
			Time += float64(dt / float32(Substeps))
		}
		Time -= float64(dt)
	}

	
	UpdateTe(te, bufferTeBig, bufferTe)
}