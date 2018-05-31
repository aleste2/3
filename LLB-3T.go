package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Heun solver for LLB equation + joule heating.
type HeunLLB3T struct{}

// Adaptive HeunLLB3T method, can be used as solver.Step
func (_ *HeunLLB3T) Step() {

	y := M.Buffer()
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)
	Hth1 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(Hth1)
	Hth2 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(Hth2)

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si * GammaLL)
	util.Assert(dt > 0)

	// stage 1

        // Rewrite to calculate m step 1 
	torqueFnLLB3T(dy0,Hth1,Hth2)
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy

  // stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)

	Time += Dt_si

        // Rewrite to calculate step 2
	torqueFnLLB3T(dy,Hth1,Hth2)

	err := cuda.MaxVecDiff(dy0, dy) * float64(dt)
	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt) //****

		// Good step, then evolve Temperatures with rk4. Equation is numericaly complicated, better to divide time step
		
		substeps:=3
		for iter:=0;iter<substeps; iter++{
			rk4Step3T(dt/float32(substeps)/float32(GammaLL))
		}

		//M.normalizeLLB()   // not in LLB!!
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./2.))
		setLastErr(err)
		setMaxTorque(dy)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(y, y, dy0, 1, -dt)  //****
		// nothing to do with temperatures now
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
	}
}

func (_ *HeunLLB3T) Free() {}
