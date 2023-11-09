package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	//		"math"
)

// Heun solver joule heating only equation. Added buffers for slow evolution
type Only2T struct {
	bufferTe    *data.Slice // buffer for slow Te evolucion
	bufferTl    *data.Slice // buffer for slow Tl evolucion
	bufferTeBig *data.Slice // buffer for Te evolution
	bufferTlBig *data.Slice // buffer for Tl evolution
}

// Adaptive HeunJH method, can be used as solver.Step
func (S2T *Only2T) Step() {

	if FixDt != 0 {
		Dt_si = FixDt
	}

	// first step ever: one-time buffer Tel init and eval
	if S2T.bufferTe == nil {
		size := Te.Mesh().Size()
		S2T.bufferTe = cuda.NewSlice(1, size)
		S2T.bufferTl = cuda.NewSlice(1, size)
		S2T.bufferTeBig = cuda.NewSlice(1, size)
		S2T.bufferTlBig = cuda.NewSlice(1, size)
		cuda.Madd2(S2T.bufferTeBig, Te.temp, Te.temp, 1, 0)
		cuda.Madd2(S2T.bufferTlBig, Tl.temp, Tl.temp, 1, 0)
	}

	dt := float32(Dt_si)
	util.Assert(dt > 0)

	// stage 1

	//NewtonStep2T(dt)
	AdaptativeNewtonStep2T(dt, S2T.bufferTe, S2T.bufferTl, S2T.bufferTeBig, S2T.bufferTlBig)
	Time += Dt_si

}

func (S2T *Only2T) Free() {
	S2T.bufferTe.Free()
	S2T.bufferTe = nil
	S2T.bufferTl.Free()
	S2T.bufferTl = nil
	S2T.bufferTeBig.Free()
	S2T.bufferTeBig = nil
	S2T.bufferTlBig.Free()
	S2T.bufferTlBig = nil
}

func NewtonStep2T(dt float32) {
	te := Te.temp
	size := Te.temp.Size()
	tl := Tl.temp
	y := M.Buffer()

	// Not needed with new Ke,Kl

	//Kel := Ke.MSlice()
	//defer Kel.Recycle()
	Cel := Ce.MSlice()
	defer Cel.Recycle()

	//Klat := Kl.MSlice()
	//defer Klat.Recycle()
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

	// Te Tl step variations
	DeltaTe := cuda.Buffer(1, size)
	defer cuda.Recycle(DeltaTe)
	DeltaTl := cuda.Buffer(1, size)
	defer cuda.Recycle(DeltaTl)

	//	cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, Kel, Cel, Klat, Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu())
	cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, kel.Gpu(), Cel, kll.Gpu(), Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
	//cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, lex21.Gpu(), Cel, kll.Gpu(), Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu(),regions.Gpu())
	err := cuda.MaxAbs(DeltaTe)
	if (err*dt < 10) || (flagOST == 1) {
		TOverstepsCounter++
		flagOST = 1
		if TOverstepsCounter >= TOversteps {
			cuda.Madd2(te, te, DeltaTe, 1, dt*float32(TOversteps)) // temp = temp + dt * dtemp0
			cuda.Madd2(tl, tl, DeltaTl, 1, dt*float32(TOversteps)) // temp = temp + dt * dtemp0
			TOverstepsCounter = 0
			flagOST = 0
		}
	} else {
		cuda.Madd2(te, te, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
		cuda.Madd2(tl, tl, DeltaTl, 1, dt) // temp = temp + dt * dtemp0
		TOverstepsCounter = 0
	}
}

func AdaptativeNewtonStep2T(dt float32, bufferTe, bufferTl, bufferTeBig, bufferTlBig *data.Slice) {

	te := Te.temp
	size := Te.temp.Size()
	tl := Tl.temp
	y := M.Buffer()

	//Kel := Ke.MSlice()
	//defer Kel.Recycle()
	Cel := Ce.MSlice()
	defer Cel.Recycle()

	//Klat := Kl.MSlice()
	//defer Klat.Recycle()
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

	// Recalculate Te,Tl
	UpdateTeTl(te, tl, bufferTeBig, bufferTe, bufferTlBig, bufferTl)
	cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, kel.Gpu(), Cel, kll.Gpu(), Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
	err := cuda.MaxAbs(DeltaTe)

	toleranceT := 1.0
	minStepTe := 0.1
	Substeps := 1
	if err*dt < float32(toleranceT) {
		if float64(err*dt) > minStepTe {
			cuda.Madd2(bufferTeBig, bufferTeBig, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
			cuda.Madd2(bufferTlBig, bufferTlBig, DeltaTl, 1, dt) // temp = temp + dt * dtemp0
		} else {
			cuda.Madd2(bufferTe, bufferTe, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
			cuda.Madd2(bufferTl, bufferTl, DeltaTl, 1, dt) // temp = temp + dt * dtemp0
			err2 := cuda.MaxAbs(bufferTe)
			// Add small buffers if needed
			if float64(err2) >= minStepTe {
				cuda.Madd2(bufferTeBig, bufferTeBig, bufferTe, 1, 1) // temp = temp + bufferTe
				cuda.Madd2(bufferTlBig, bufferTlBig, bufferTl, 1, 1) // temp = temp + bufferTl
				cuda.Madd2(bufferTe, bufferTe, DeltaTe, 0, 0)        // bufferTe to 0
				cuda.Madd2(bufferTl, bufferTl, DeltaTl, 0, 0)        // bufferTl to 0
			}
		}
	} else {
		for {
			Substeps = Substeps * 2
			if (err * dt / float32(Substeps)) < float32(toleranceT) {
				break
			}
		}
		for iSubsteps := 0; iSubsteps < Substeps; iSubsteps++ {
			//cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, kel.Gpu(), Cel, kll.Gpu(), Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
			//cuda.Madd2(te, te, DeltaTe, 1, dt/float32(Substeps)) // temp = temp + dt * dtemp0
			//cuda.Madd2(tl, tl, DeltaTl, 1, dt/float32(Substeps)) // temp = temp + dt * dtemp0
			UpdateTeTl(te, tl, bufferTeBig, bufferTe, bufferTlBig, bufferTl)
			cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, kel.Gpu(), Cel, kll.Gpu(), Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu(), regions.Gpu())
			cuda.Madd2(bufferTeBig, bufferTeBig, DeltaTe, 1, dt/float32(Substeps)) // temp = temp + dt * dtemp0
			cuda.Madd2(bufferTlBig, bufferTlBig, DeltaTl, 1, dt/float32(Substeps)) // temp = temp + dt * dtemp0
			Time += float64(dt / float32(Substeps))
		}
		Time -= float64(dt)
	}
	UpdateTeTl(te, tl, bufferTeBig, bufferTe, bufferTlBig, bufferTl)
}

// Update Te and Tl with buffers
func UpdateTeTl(Te, Tl, bufferTeBig, bufferTe, bufferTlBig, bufferTl *data.Slice) {
	cuda.Madd2(Te, bufferTe, bufferTeBig, 1, 1)
	cuda.Madd2(Tl, bufferTl, bufferTlBig, 1, 1)
}
