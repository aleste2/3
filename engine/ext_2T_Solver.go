package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	//		"math"
)

// Heun solver joule heating only equation.
type Only2T struct {
	bufferTe *data.Slice // buffer for slow Te evolucion
	bufferTl *data.Slice // buffer for slow Tl evolucion
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
	}

	dt := float32(Dt_si)
	util.Assert(dt > 0)

	// stage 1

	//NewtonStep2T(dt)
	AdaptativeNewtonStep2T(dt, S2T.bufferTe, S2T.bufferTl)
	Time += Dt_si
}

func (S2T *Only2T) Free() {
	S2T.bufferTe.Free()
	S2T.bufferTe = nil
	S2T.bufferTl.Free()
	S2T.bufferTl = nil
}

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

	cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, Kel, Cel, Klat, Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu())
	err := cuda.MaxAbs(DeltaTe)
	if (err*dt < 10) || (flagOST == 1) {
		TOverstepsCounter++
		flagOST = 1
		if TOverstepsCounter >= TOversteps {
			cuda.Madd2(te, te, DeltaTe, 1, dt*float32(TOversteps)) // temp = temp + dt * dtemp0
			cuda.Madd2(tl, tl, DeltaTl, 1, dt*float32(TOversteps)) // temp = temp + dt * dtemp0
			TOverstepsCounter = 0
			flagOST = 0
			//print("si\n")
		}
	} else {
		cuda.Madd2(te, te, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
		cuda.Madd2(tl, tl, DeltaTl, 1, dt) // temp = temp + dt * dtemp0
		TOverstepsCounter = 0
	}

	//setLastErr(float64(err))
	//Time += Dt_si

	//cuda.Evaldt02T(te,t0e,tl,t0l,y,Kel,Cel,Klat,Clat,Gellat,Dth,Tsubsth,Tausubsth,res,Qext,CD,j,M.Mesh())
	//cuda.Madd3(te, te, t0e, t0e, 1, 0.5*dt, -0.5*dt) //****
	//cuda.Madd3(tl, tl, t0l, t0l, 1, 0.5*dt, -0.5*dt) //****

}

func Adaptative2NewtonStep2T(dt float32, bufferTe, bufferTl *data.Slice) {
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

	cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, Kel, Cel, Klat, Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu())
	err := cuda.MaxAbs(DeltaTe)

	toleranceT := 1.0
	minStepTe := 0.1
	Substeps := 1
	//ScaleNoiseLLB=float64(err*dt)
	if err*dt < float32(toleranceT) {
		cuda.Madd2(bufferTe, bufferTe, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
		cuda.Madd2(bufferTl, bufferTl, DeltaTl, 1, dt) // temp = temp + dt * dtemp0
	} else {
		for {
			Substeps = Substeps * 2
			if (err * dt / float32(Substeps)) < float32(toleranceT) {
				break
			}
		}
		for iSubsteps := 0; iSubsteps < Substeps; iSubsteps++ {
			cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, Kel, Cel, Klat, Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu())
			cuda.Madd2(te, te, DeltaTe, 1, dt/float32(Substeps)) // temp = temp + dt * dtemp0
			cuda.Madd2(tl, tl, DeltaTl, 1, dt/float32(Substeps)) // temp = temp + dt * dtemp0
			Time += float64(dt / float32(Substeps))
		}
		Time -= float64(dt)
	}

	// Empty Buffers
	err2 := cuda.MaxAbs(bufferTe)
	if float64(err2) >= minStepTe {
		cuda.Madd2(te, te, bufferTe, 1, 1)            // temp = temp + bufferTe
		cuda.Madd2(tl, tl, bufferTl, 1, 1)            // temp = temp + bufferTl
		cuda.Madd2(bufferTe, bufferTe, DeltaTe, 0, 0) // bufferTe to 0
		cuda.Madd2(bufferTl, bufferTl, DeltaTl, 0, 0) // bufferTl to 0
	}

}

func AdaptativeNewtonStep2T(dt float32, bufferTe, bufferTl *data.Slice) {
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

	cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, Kel, Cel, Klat, Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu())
	err := cuda.MaxAbs(DeltaTe)

	toleranceT := 1.0
	minStepTe := 0.1
	Substeps := 1
	//ScaleNoiseLLB=float64(err*dt)
	if err*dt < float32(toleranceT) {
		if float64(err*dt) > minStepTe {
			cuda.Madd2(te, te, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
			cuda.Madd2(tl, tl, DeltaTl, 1, dt) // temp = temp + dt * dtemp0
			//print(Time," Directo\n")
		} else {
			//			print(Time," Buffer\n")
			cuda.Madd2(bufferTe, bufferTe, DeltaTe, 1, dt) // temp = temp + dt * dtemp0
			cuda.Madd2(bufferTl, bufferTl, DeltaTl, 1, dt) // temp = temp + dt * dtemp0
			err2 := cuda.MaxAbs(bufferTe)
			// Add small buffers if needed
			if float64(err2) >= minStepTe {
				//			print(Time," Empty Buffer\n")
				cuda.Madd2(te, te, bufferTe, 1, 1)            // temp = temp + bufferTe
				cuda.Madd2(tl, tl, bufferTl, 1, 1)            // temp = temp + bufferTl
				cuda.Madd2(bufferTe, bufferTe, DeltaTe, 0, 0) // bufferTe to 0
				cuda.Madd2(bufferTl, bufferTl, DeltaTl, 0, 0) // bufferTl to 0
			}
		}
	} else {
		for {
			Substeps = Substeps * 2
			//							print(Time," Multiply Substeps\n")
			//cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, Kel, Cel, Klat, Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu())
			//err := cuda.MaxAbs(DeltaTe)
			if (err * dt / float32(Substeps)) < float32(toleranceT) {
				break
			}
		}
		for iSubsteps := 0; iSubsteps < Substeps; iSubsteps++ {
			cuda.Evaldt02T(te, DeltaTe, tl, DeltaTl, y, Kel, Cel, Klat, Clat, Gellat, Dth, Tsubsth, Tausubsth, res, Qext, CD, j, M.Mesh(), geometry.Gpu())
			cuda.Madd2(te, te, DeltaTe, 1, dt/float32(Substeps)) // temp = temp + dt * dtemp0
			cuda.Madd2(tl, tl, DeltaTl, 1, dt/float32(Substeps)) // temp = temp + dt * dtemp0
			Time += float64(dt / float32(Substeps))
			//ScaleNoiseLLB=float64(err*dt/float32(Substeps))
			//										print(Time," Calculus Substeps\n")
		}
		//print(Substeps,"\n")
		Time -= float64(dt)
	}
	//TSubsteps=Substeps

}
