package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Euler M3TM + joule heating + 2T model.
type EulerM3TMs struct {
}

// Adaptive Euler M3TM method, can be used as solver.Step
func (EulerM3TM *EulerM3TMs) Step() {

	y1 := M.Buffer()

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
	AdaptativeFTCSStep2T(float32(Dt_si))
	musStep(dy11, dy1)
	TTMplus(dy11, dy1, Te.temp, float32(Dt_si))
	cuda.Madd3(y1, y1, dy11, dy1, 1, 0.5*dt, -0.5*dt) //****

}

func (EulerM3TM *EulerM3TMs) Free() {
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
