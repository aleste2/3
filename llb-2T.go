package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"math"
)

// Heun solver for LLB equation + joule heating + 2T model.
type HeunLLB2T struct{}

// Adaptive HeunLLB2T method, can be used as solver.Step
func (_ *HeunLLB2T) Step() {

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
	torqueFnLLB2T(dy0,Hth1,Hth2)
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy

        

	// stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)

	Time += Dt_si

        // Rewrite to calculate step 2
	torqueFnLLB2T(dy,Hth1,Hth2)

	err := cuda.MaxVecDiff(dy0, dy) * float64(dt)
	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt) //****

		// Good step, then evolve Temperatures with rk4. Equation is numericaly complicated, better to divide time step
		
		substeps:=3
		for iter:=0;iter<substeps; iter++{
			rk4Step2T(dt/float32(substeps)/float32(GammaLL))
		}

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

func (_ *HeunLLB2T) Free() {}



func rk4Step2T(dt float32) {
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
	t0e := cuda.Buffer(1, size)
	defer cuda.Recycle(t0e)
	data.Copy(t0e, te)
	t0l := cuda.Buffer(1, size)
	defer cuda.Recycle(t0l)
	data.Copy(t0l, tl)

	k1e, k2e, k3e, k4e := cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size)
	k1l, k2l, k3l, k4l := cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size)

	defer cuda.Recycle(k1e)
	defer cuda.Recycle(k2e)
	defer cuda.Recycle(k3e)
	defer cuda.Recycle(k4e)
	defer cuda.Recycle(k1l)
	defer cuda.Recycle(k2l)
	defer cuda.Recycle(k3l)
	defer cuda.Recycle(k4l)

	// stage 1
        cuda.Evaldt02T(te,k1e,tl,k1l,y,Kel,Cel,Klat,Clat,Gellat,Dth,Tsubsth,Tausubsth,res,Qext,CD,j,M.Mesh())

	// stage 2
	cuda.Madd2(te, te, k1e, 1, (1./2.)*dt) // m = m*1 + k1*h/2
	cuda.Madd2(tl, tl, k1l, 1, (1./2.)*dt) // m = m*1 + k1*h/2

        cuda.Evaldt02T(te,k2e,tl,k2l,y,Kel,Cel,Klat,Clat,Gellat,Dth,Tsubsth,Tausubsth,res,Qext,CD,j,M.Mesh())

	// stage 3
	cuda.Madd2(te, t0e, k2e, 1, (1./2.)*dt) // m = m0*1 + k2*1/2
	cuda.Madd2(tl, t0l, k2l, 1, (1./2.)*dt) // m = m0*1 + k2*1/2

        cuda.Evaldt02T(te,k3e,tl,k3l,y,Kel,Cel,Klat,Clat,Gellat,Dth,Tsubsth,Tausubsth,res,Qext,CD,j,M.Mesh())

	// stage 4
	cuda.Madd2(te, t0e, k3e, 1, 1.*dt) // m = m0*1 + k3*1
	cuda.Madd2(tl, t0l, k3l, 1, 1.*dt) // m = m0*1 + k3*1

        cuda.Evaldt02T(te,k4e,tl,k4l,y,Kel,Cel,Klat,Clat,Gellat,Dth,Tsubsth,Tausubsth,res,Qext,CD,j,M.Mesh())

	madd5(te, t0e, k1e, k2e, k3e, k4e, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
	madd5(tl, t0l, k1l, k2l, k3l, k4l, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)

}
