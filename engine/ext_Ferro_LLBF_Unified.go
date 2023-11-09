package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Heun solver for Ferro LLB equation + joule heating + 2T model.
type HeunLLBFerroUnified struct {
	bufferTe    *data.Slice // buffer for slow Te evolucion
	bufferTl    *data.Slice // buffer for slow Tl evolucion
	bufferTeBig *data.Slice // buffer for slow Te evolucion
	bufferTlBig *data.Slice // buffer for slow Tl evolucion
}

// Adaptive HeunLLB2T method, can be used as solver.Step
func (LLB *HeunLLBFerroUnified) Step() {

	y1 := M.Buffer()

	// Temperature Buffers
	if LLB2Tf == true {
		if LLB.bufferTe == nil {
			size := Te.Mesh().Size()
			LLB.bufferTe = cuda.NewSlice(1, size)
			LLB.bufferTl = cuda.NewSlice(1, size)
			LLB.bufferTeBig = cuda.NewSlice(1, size)
			LLB.bufferTlBig = cuda.NewSlice(1, size)
			cuda.Madd2(LLB.bufferTe, LLB.bufferTe, LLB.bufferTe, 0, 0) // bufferTe to 0
			cuda.Madd2(LLB.bufferTl, LLB.bufferTl, LLB.bufferTl, 0, 0) // bufferTl to 0
			cuda.Madd2(LLB.bufferTeBig, Te.temp, Te.temp, 1, 0)
			cuda.Madd2(LLB.bufferTlBig, Tl.temp, Tl.temp, 1, 0)
		}
	}

	// For renorm
	y01 := cuda.Buffer(VECTOR, y1.Size())
	defer cuda.Recycle(y01)

	cuda.Madd2(y01, y1, y01, 1, 0) // y = y + dt * dy
	//

	dy1 := cuda.Buffer(VECTOR, y1.Size())
	defer cuda.Recycle(dy1)

	Hth1a := cuda.Buffer(VECTOR, y1.Size())
	defer cuda.Recycle(Hth1a)
	Hth2a := cuda.Buffer(VECTOR, y1.Size())
	defer cuda.Recycle(Hth2a)

	cuda.Zero(Hth1a)
	cuda.Zero(Hth2a)
	if JHThermalnoise == true {
		if (LLB2Tf == true) || (LLBJHf == true) {
			B_therm.LLBAddTo(Hth1a)
			B_therm.LLBAddTo(Hth2a)
		} else {
			B_therm.AddTo(Hth1a)
			B_therm.AddTo(Hth2a)
		}
	}

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si * GammaLL)
	util.Assert(dt > 0)

	// stage 1

	// Rewrite to calculate m step 1
	torqueFnFerroLLBUnified(dy1, Hth1a, Hth2a)
	cuda.Madd2(y1, y1, dy1, 1, dt) // y = y + dt * dy

	// stage 2
	dy11 := cuda.Buffer(3, y1.Size())
	defer cuda.Recycle(dy11)
	Time += Dt_si

	// Rewrite to calculate step 2
	torqueFnFerroLLBUnified(dy11, Hth1a, Hth2a)

	err := cuda.MaxVecDiff(dy1, dy11) * float64(dt)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y1, y1, dy11, dy1, 1, 0.5*dt, -0.5*dt) //****

		if LLB2Tf == true {
			//for iter := 0; iter < TSubsteps; iter++ {
			//				NewtonStep2T(float32(Dt_si) / float32(TSubsteps))
			AdaptativeNewtonStep2T(float32(Dt_si), LLB.bufferTe, LLB.bufferTl, LLB.bufferTeBig, LLB.bufferTlBig)
			//}
		}

		if LLBJHf == true {
			//			StepJH(float32(Dt_si))

			if TOversteps == 1 {
				StepJH(float32(Dt_si))
			}
			if TOversteps > 1 {
				TOverstepsCounter++
				if TOverstepsCounter >= TOversteps {
					StepJH(float32(Dt_si * float64(TOversteps)))
					TOverstepsCounter = 0
				}
			}

		}

		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./2.))
		setLastErr(err)
		setMaxTorque(dy1)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(y1, y1, dy1, 1, -dt) //****
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
	}
}

func (LLB *HeunLLBFerroUnified) Free() {
	LLB.bufferTe.Free()
	LLB.bufferTe = nil
	LLB.bufferTl.Free()
	LLB.bufferTl = nil
	LLB.bufferTeBig.Free()
	LLB.bufferTeBig = nil
	LLB.bufferTlBig.Free()
	LLB.bufferTlBig = nil
}

// Torque for antiferro LLB 2T

// write torque to dst and increment NEvals
// Now everything here to use 2 lattices at the same time
func torqueFnFerroLLBUnified(dst1 *data.Slice, hth1a, hth2a *data.Slice) {

	// Set Effective field
	SetDemagField(dst1)
	AddExchangeField(dst1)
	AddAnisotropyField(dst1)
	//AddAFMExchangeField(dst)  // AFM Exchange non adjacent layers
	B_ext.AddTo(dst1)
	AddCustomField(dst1)

	// Add to sublattice 1 and 2
	alpha := Alpha.MSlice()
	defer alpha.Recycle()

	Tcurie := TCurie.MSlice()
	defer Tcurie.Recycle()
	Msat := Msat.MSlice()
	defer Msat.Recycle()
	temp := Temp.MSlice()
	defer temp.Recycle()

	NV := nv.MSlice()
	defer NV.Recycle()

	MU1 := mu1.MSlice()
	defer MU1.Recycle()

	J0AA := J0aa.MSlice()
	defer J0AA.Recycle()

	// Direct deltai induction
	deltam := deltaM.MSlice()
	defer deltam.Recycle()
	Qext := Qext.MSlice()
	defer Qext.Recycle()

	TTM := 0
	if (LLB2Tf == true) || (LLBJHf == true) {
		TTM = 1
	}
	if Precess {
		cuda.LLBTorqueFerroUnified(dst1, M.Buffer(), dst1, temp, Te.temp, alpha, Tcurie, Msat, hth1a, hth2a, NV, MU1, J0AA, Qext, deltam, TTM, geometry.Gpu()) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst1, M.Buffer(), dst1)
	}

	// STT
	AddSTTorque(dst1)

	FreezeSpins(dst1)
	NEvals++
}
