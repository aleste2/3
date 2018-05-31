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
        //cuda.Evaldt03T(temp0e,dtemp0e,temp0l,dtemp0l,temp0s,dtemp0s,y,Kel,Cel,Klat,Clat,Ksp,Csp,Gellat,Gelsp,Glatsp,Dth,Tsubsth,Tausubsth,res,Qext,j,M.Mesh())
	//cuda.Madd2(temp0e, temp0e, dtemp0e, 1, dt/float32(GammaLL)) // temp = temp + dt * dtemp0 electron
	//cuda.Madd2(temp0l, temp0l, dtemp0l, 1, dt/float32(GammaLL)) // temp = temp + dt * dtemp0 lattice
	//cuda.Madd2(temp0s, temp0s, dtemp0s, 1, dt/float32(GammaLL)) // temp = temp + dt * dtemp0 spins
        

	// stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)

	//dtempe := cuda.Buffer(1, temp0e.Size())
	//defer cuda.Recycle(dtempe)
	//dtempl := cuda.Buffer(1, temp0l.Size())
	//defer cuda.Recycle(dtempl)
	//dtemps := cuda.Buffer(1, temp0s.Size())
	//defer cuda.Recycle(dtemps)
	Time += Dt_si

        // Rewrite to calculate step 2
	torqueFnLLB3T(dy,Hth1,Hth2)
        //cuda.Evaldt03T(temp0e,dtempe,temp0l,dtempl,temp0s,dtemps,y,Kel,Cel,Klat,Clat,Ksp,Csp,Gellat,Gelsp,Glatsp,Dth,Tsubsth,Tausubsth,res,Qext,j,M.Mesh())

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

		//cuda.Madd3(temp0e, temp0e, dtempe, dtemp0e, 1, 0.5*dt/float32(GammaLL), -0.5*dt/float32(GammaLL)) //****
		//cuda.Madd3(temp0l, temp0l, dtempl, dtemp0l, 1, 0.5*dt/float32(GammaLL), -0.5*dt/float32(GammaLL)) //****
		//cuda.Madd3(temp0s, temp0s, dtemps, dtemp0s, 1, 0.5*dt/float32(GammaLL), -0.5*dt/float32(GammaLL)) //****
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
		//cuda.Madd2(temp0e, temp0e, dtemp0e, 1, -dt/float32(GammaLL)) // temp = temp - dt * dtemp0
		//cuda.Madd2(temp0l, temp0l, dtemp0l, 1, -dt/float32(GammaLL)) // temp = temp - dt * dtemp0
		//cuda.Madd2(temp0s, temp0s, dtemp0s, 1, -dt/float32(GammaLL)) // temp = temp - dt * dtemp0
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
	}
}

func (_ *HeunLLB3T) Free() {}



func rk4Step3T(dt float32) {
	te := Te.temp
	size := Te.temp.Size()
	tl := Tl.temp
	ts := Ts.temp
	y := M.Buffer()

	Kel := Ke.MSlice()
	defer Kel.Recycle()
	Cel := Ce.MSlice()
	defer Cel.Recycle()

	Klat := Kl.MSlice()
	defer Klat.Recycle()
	Clat := Cl.MSlice()
	defer Clat.Recycle()

	Ksp := Ks.MSlice()
	defer Ksp.Recycle()
	Csp := Cs.MSlice()
	defer Csp.Recycle()

	Gellat := Gel.MSlice()
	defer Gellat.Recycle()
	Gelsp := Ges.MSlice()
	defer Gelsp.Recycle()
	Glatsp := Gls.MSlice()
	defer Glatsp.Recycle()

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
	t0e := cuda.Buffer(1, size)
	defer cuda.Recycle(t0e)
	data.Copy(t0e, te)
	t0l := cuda.Buffer(1, size)
	defer cuda.Recycle(t0l)
	data.Copy(t0l, tl)
	t0s := cuda.Buffer(1, size)
	defer cuda.Recycle(t0s)
	data.Copy(t0s, ts)

	k1e, k2e, k3e, k4e := cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size)
	k1l, k2l, k3l, k4l := cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size)
	k1s, k2s, k3s, k4s := cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size), cuda.Buffer(1, size)

	defer cuda.Recycle(k1e)
	defer cuda.Recycle(k2e)
	defer cuda.Recycle(k3e)
	defer cuda.Recycle(k4e)
	defer cuda.Recycle(k1l)
	defer cuda.Recycle(k2l)
	defer cuda.Recycle(k3l)
	defer cuda.Recycle(k4l)
	defer cuda.Recycle(k1s)
	defer cuda.Recycle(k2s)
	defer cuda.Recycle(k3s)
	defer cuda.Recycle(k4s)

	// stage 1
        cuda.Evaldt03T(te,k1e,tl,k1l,ts,k1s,y,Kel,Cel,Klat,Clat,Ksp,Csp,Gellat,Gelsp,Glatsp,Dth,Tsubsth,Tausubsth,res,Qext,j,M.Mesh())

	// stage 2
	cuda.Madd2(te, te, k1e, 1, (1./2.)*dt) // m = m*1 + k1*h/2
	cuda.Madd2(tl, tl, k1l, 1, (1./2.)*dt) // m = m*1 + k1*h/2
	cuda.Madd2(ts, ts, k1s, 1, (1./2.)*dt) // m = m*1 + k1*h/2

        cuda.Evaldt03T(te,k2e,tl,k2l,ts,k2s,y,Kel,Cel,Klat,Clat,Ksp,Csp,Gellat,Gelsp,Glatsp,Dth,Tsubsth,Tausubsth,res,Qext,j,M.Mesh())

	// stage 3
	cuda.Madd2(te, t0e, k2e, 1, (1./2.)*dt) // m = m0*1 + k2*1/2
	cuda.Madd2(tl, t0l, k2l, 1, (1./2.)*dt) // m = m0*1 + k2*1/2
	cuda.Madd2(ts, t0s, k2s, 1, (1./2.)*dt) // m = m0*1 + k2*1/2

        cuda.Evaldt03T(te,k3e,tl,k3l,ts,k3s,y,Kel,Cel,Klat,Clat,Ksp,Csp,Gellat,Gelsp,Glatsp,Dth,Tsubsth,Tausubsth,res,Qext,j,M.Mesh())

	// stage 4
	cuda.Madd2(te, t0e, k3e, 1, 1.*dt) // m = m0*1 + k3*1
	cuda.Madd2(tl, t0l, k3l, 1, 1.*dt) // m = m0*1 + k3*1
	cuda.Madd2(ts, t0s, k3s, 1, 1.*dt) // m = m0*1 + k3*1

        cuda.Evaldt03T(te,k4e,tl,k4l,ts,k4s,y,Kel,Cel,Klat,Clat,Ksp,Csp,Gellat,Gelsp,Glatsp,Dth,Tsubsth,Tausubsth,res,Qext,j,M.Mesh())

	madd5(te, t0e, k1e, k2e, k3e, k4e, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
	madd5(tl, t0l, k1l, k2l, k3l, k4l, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)
	madd5(ts, t0s, k1s, k2s, k3s, k4s, 1, (1./6.)*dt, (1./3.)*dt, (1./3.)*dt, (1./6.)*dt)

}
