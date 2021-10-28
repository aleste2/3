package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

var (
	tauOSC = NewScalarParam("tauOSC", "s", "lectron recombination time")
	ROSC   = NewScalarExcitation("ROSC", "s", "Spin injection rate")
	JexOSC = NewScalarParam("JexOSC", "J", "Coupling OSC with m")
	dirOSC = NewVectorParam("dirOSC", "", "Direction of light")
	SOSC   magnetization // reduced magnetization (unit length)
	B_OSC  = NewVectorField("B_OSC", "T", "Optical spin field", AddOSCField)
)

func init() {
	DeclLValue("SOSC", &SOSC, `Optical spin current`)
	SOSC.name = "s_"
	DeclFunc("InitOSC", InitOSC, "Init Optical spin currents")
}

func InitOSC() {
	SOSC.alloc()
}

type HeunLLB2TOSC struct{}

// Adaptive HeunLLB2T method, can be used as solver.Step
func (_ *HeunLLB2TOSC) Step() {

	y := M.Buffer()
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)
	Hth1 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(Hth1)
	Hth2 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(Hth2)

	cuda.Zero(Hth1)
	B_therm.LLBAddTo(Hth1)
	cuda.Zero(Hth2)
	B_therm.LLBAddTo(Hth2)

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si * GammaLL)
	util.Assert(dt > 0)

	// stage 1

	// Rewrite to calculate m step 1
	torqueFnLLB2T(dy0, Hth1, Hth2)
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy

	// stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)

	Time += Dt_si

	// Rewrite to calculate step 2
	torqueFnLLB2T(dy, Hth1, Hth2)

	err := cuda.MaxVecDiff(dy0, dy) * float64(dt)
	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt) //****

		// Good step, then evolve Temperatures with rk4. Equation is numericaly complicated, better to divide time step

		//Time -= Dt_si
		for iter := 0; iter < TSubsteps; iter++ {
			NewtonStep2T(float32(Dt_si) / float32(TSubsteps))
			StepOST(float32(Dt_si) / float32(TSubsteps))
			//Time += Dt_si / float64(TSubsteps)
		}
		/*Time -= Dt_si
		for iter := 0; iter < 10; iter++ {
			StepOST(float32(Dt_si)/10)
			Time += Dt_si/10
		}*/
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./2.))
		setLastErr(err)
		setMaxTorque(dy)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(y, y, dy0, 1, -dt) //****
		// nothing to do with temperatures now
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
	}
}

func (_ *HeunLLB2TOSC) Free() {}

func StepOST(dt float32) {

	y := M.Buffer()
	S := SOSC.Buffer()
	size := S.Size()
	dy1 := cuda.Buffer(VECTOR, size)
	defer cuda.Recycle(dy1)

	tau := tauOSC.MSlice()
	defer tau.Recycle()
	Jex := JexOSC.MSlice()
	defer Jex.Recycle()
	R := ROSC.MSlice()
	defer R.Recycle()
	dir := dirOSC.MSlice()
	defer dir.Recycle()

	//cuda.StepOST(y, S, dy1, tau, Jex, R, dir, M.Mesh())
	//cuda.Madd2(S, S, dy1, 1, dt) // temp = temp + dt * dtemp0

	//err :=cuda.MaxVecNorm(dy1)
	//print(err,"\n")

	/*factor:=1e12
	if (err>factor) {
		steps:=int(err/factor)
		print(err, " ",steps,"\n")
		for i:=0;i<steps;i++ {
			cuda.StepOST(y,S,dy1,tau,Jex,R,dir, M.Mesh())
	  	cuda.Madd2(S, S, dy1, 1, float32(dt/float32(steps))) // temp = temp + dt * dtemp0
		}
	}	else {
		cuda.Madd2(S, S, dy1, 1, dt) // temp = temp + dt * dtemp0
	}*/

	// RK4 Step

	// backup magnetization
	S0 := cuda.Buffer(3, size)
	defer cuda.Recycle(S0)
	data.Copy(S0, S)

	dy1, dy2, dy3, dy4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(dy1)
	defer cuda.Recycle(dy2)
	defer cuda.Recycle(dy3)
	defer cuda.Recycle(dy4)

	h := float32(dt) // time step

	// stage 1
	cuda.StepOST(y, S, dy1, tau, Jex, R, dir, M.Mesh())

	// stage 2
	cuda.Madd2(S, S, dy1, 1, (1./2.)*h) // m = m*1 + k1*h/2
	cuda.StepOST(y, S, dy2, tau, Jex, R, dir, M.Mesh())

	// stage 3
	cuda.Madd2(S, S0, dy2, 1, (1./2.)*h) // m = m0*1 + k2*1/2
	cuda.StepOST(y, S, dy3, tau, Jex, R, dir, M.Mesh())

	// stage 4
	cuda.Madd2(S, S0, dy3, 1, 1.*h) // m = m0*1 + k3*1
	cuda.StepOST(y, S, dy4, tau, Jex, R, dir, M.Mesh())

	cuda.Madd5(S, S0, dy1, dy2, dy3, dy4, 1, (1./6.)*h, (1./3.)*h, (1./3.)*h, (1./6.)*h)

}

func AddOSCField(dst *data.Slice) {

	ms := Msat.MSlice()
	defer ms.Recycle()
	Jex := JexOSC.MSlice()
	defer Jex.Recycle()
	cuda.AddOSTField(dst, SOSC.Buffer(), ms, Jex)

}
