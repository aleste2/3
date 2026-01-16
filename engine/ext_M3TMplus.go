package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Euler/Heun M3TM + joule heating + 2T model.
type EulerM3TMs struct {
	bufferTe    *data.Slice // buffer for slow Te evolucion
	bufferTl    *data.Slice // buffer for slow Tl evolucion
	bufferTeBig *data.Slice // buffer for slow Te evolucion
	bufferTlBig *data.Slice // buffer for slow Tl evolucion
}

// Adaptive Euler M3TM method, can be used as solver.Step
func (EulerM3TM *EulerM3TMs) Step() {

	y1 := M.Buffer()

	// Temperature Buffers
	if LLB2Tf == true {
		if EulerM3TM.bufferTe == nil {
			size := Te.Mesh().Size()
			EulerM3TM.bufferTe = cuda.NewSlice(1, size)
			EulerM3TM.bufferTl = cuda.NewSlice(1, size)
			EulerM3TM.bufferTeBig = cuda.NewSlice(1, size)
			EulerM3TM.bufferTlBig = cuda.NewSlice(1, size)
			cuda.Madd2(EulerM3TM.bufferTe, EulerM3TM.bufferTe, EulerM3TM.bufferTe, 0, 0) // bufferTe to 0
			cuda.Madd2(EulerM3TM.bufferTl, EulerM3TM.bufferTl, EulerM3TM.bufferTl, 0, 0) // bufferTl to 0
			cuda.Madd2(EulerM3TM.bufferTeBig, Te.temp, Te.temp, 1, 0)
			cuda.Madd2(EulerM3TM.bufferTlBig, Tl.temp, Tl.temp, 1, 0)
		}
	}

	if LLBJHf == true {
		if EulerM3TM.bufferTe == nil {
			size := Te.Mesh().Size()
			EulerM3TM.bufferTe = cuda.NewSlice(1, size)
			EulerM3TM.bufferTeBig = cuda.NewSlice(1, size)
			cuda.Madd2(EulerM3TM.bufferTe, EulerM3TM.bufferTe, EulerM3TM.bufferTe, 0, 0) // bufferTe to 0
			cuda.Madd2(EulerM3TM.bufferTeBig, Te.temp, Te.temp, 1, 0)
		}
	}
	Dt_si = FixDt
	dt := float32(Dt_si * GammaLL)

	dy1 := cuda.Buffer(VECTOR, y1.Size())
	defer cuda.Recycle(dy1)

	// Rewrite to calculate m step 1
	m3TMtorque(dy1)
	cuda.Madd2(y1, y1, dy1, 1, dt) // y = y + dt * dy

	// stage 2
	dy11 := cuda.Buffer(3, y1.Size())
	defer cuda.Recycle(dy11)
	Time += Dt_si
	m3TMtorque(dy11)

	err := cuda.MaxVecDiff(dy1, dy11) * float64(Dt_si)
	setLastErr(err)
	setMaxTorque(dy1)
	AdaptativeNewtonStep2T(float32(Dt_si), EulerM3TM.bufferTe, EulerM3TM.bufferTl, EulerM3TM.bufferTeBig, EulerM3TM.bufferTlBig)
	musStep(dy11, dy1)
	//TTMplus(dy11, dy1, EulerM3TM.bufferTe, EulerM3TM.bufferTeBig, float32(Dt_si/1.7595e11))
	cuda.Madd3(y1, y1, dy11, dy1, 1, 0.5*dt, -0.5*dt) //****

	/*
		dy1 := cuda.Buffer(VECTOR, y1.Size())
		defer cuda.Recycle(dy1)

		if FixDt != 0 {
			Dt_si = FixDt
		}

		dt := float32(Dt_si * GammaLL)
		util.Assert(dt > 0)

		// stage 1

		// Rewrite to calculate m step 1
		//m3TMtorque(dy1)
		//cuda.Madd2(y1, y1, dy1, 1, float32(Dt_si)) // y = y + dt * dy

		// stage 2
		dy11 := cuda.Buffer(3, y1.Size())
		defer cuda.Recycle(dy11)
		Time += Dt_si

		// Rewrite to calculate step 2
		//m3TMtorque(dy11)
		err := cuda.MaxVecDiff(dy1, dy11) * float64(dt)
		// adjust next time step
		if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
			// step OK
			if sdf == true {
				//musStep(dy11, dy1)
				//TTMplus(dy11, dy1, EulerM3TM.bufferTe, EulerM3TM.bufferTeBig, float32(Dt_si/1.7595e11))
			}
			//cuda.Madd3(y1, y1, dy11, dy1, 1, 0.5*float32(Dt_si), -0.5*float32(Dt_si)) //****
			if LLB2Tf == true {
				AdaptativeNewtonStep2T(float32(Dt_si), EulerM3TM.bufferTe, EulerM3TM.bufferTl, EulerM3TM.bufferTeBig, EulerM3TM.bufferTlBig)
			}
			//if sdf == true { // Moment contribution
			//MusMoment(float32(float32(Dt_si))) // dt in SI units
			//}
			if LLBJHf == true {
				AdaptativeNewtonStepJH(float32(Dt_si), EulerM3TM.bufferTe, EulerM3TM.bufferTeBig)
			}
			NSteps++
			adaptDt(math.Pow(MaxErr/err, 1./2.))
			setLastErr(err)
			setMaxTorque(dy1)
		} else {
			// undo bad step
			util.Assert(FixDt == 0)
			Time -= Dt_si
			//cuda.Madd2(y1, y1, dy1, 1, -dt) //****
			NUndone++
			adaptDt(math.Pow(MaxErr/err, 1./3.))
		}
	*/
}

func (EulerM3TM *EulerM3TMs) Free() {
	EulerM3TM.bufferTe.Free()
	EulerM3TM.bufferTe = nil
	EulerM3TM.bufferTl.Free()
	EulerM3TM.bufferTl = nil
	EulerM3TM.bufferTeBig.Free()
	EulerM3TM.bufferTeBig = nil
	EulerM3TM.bufferTlBig.Free()
	EulerM3TM.bufferTlBig = nil
}

// Torque for M3TM

func m3TMtorque(dst1 *data.Slice) {
	// Add to sublattice 1 and 2
	Tcurie := TCurie.MSlice()
	defer Tcurie.Recycle()

	Tausd := tausd.MSlice()
	defer Tausd.Recycle()

	cuda.M3TMtorque(dst1, M.Buffer(), mus.Buffer(), Te.temp, Tcurie, Tausd) // overwrite dst with torque
}
