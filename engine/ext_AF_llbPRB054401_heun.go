package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/util"
	"github.com/mumax/3/data"
	"math"
)


// Heun solver for LLB equation.
type HeunAFLLBPRB054401 struct{}

// Adaptive HeunLLB method, can be used as solver.Step
func (_ *HeunAFLLBPRB054401) Step() {

	y1 := M1.Buffer()
	y2 := M2.Buffer()

	dy1 := cuda.Buffer(VECTOR, y1.Size())
	defer cuda.Recycle(dy1)
	dy2 := cuda.Buffer(VECTOR, y1.Size())
	defer cuda.Recycle(dy2)

	Hth1a := cuda.Buffer(VECTOR, y1.Size())
	defer cuda.Recycle(Hth1a)
	Hth2a := cuda.Buffer(VECTOR, y1.Size())
	defer cuda.Recycle(Hth2a)
	Hth1b := cuda.Buffer(VECTOR, y1.Size())
	defer cuda.Recycle(Hth1b)
	Hth2b := cuda.Buffer(VECTOR, y1.Size())
	defer cuda.Recycle(Hth2b)

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si * GammaLL)
	dt1 := float32(Dt_si * GammaLL1)
	dt2 := float32(Dt_si * GammaLL2)

	util.Assert(dt > 0)

	// stage 1

  // Rewrite to calculate m step 1 
	torqueFnAFLLBPRB(dy1,dy2,Hth1a,Hth2a,Hth1b,Hth2b)
	cuda.Madd2(y1, y1, dy1, 1, dt1) // y = y + dt * dy
	cuda.Madd2(y2, y2, dy2, 1, dt2) // y = y + dt * dy


	// stage 2
	dy11 := cuda.Buffer(3, y1.Size())
	defer cuda.Recycle(dy11)
	dy12 := cuda.Buffer(3, y1.Size())
	defer cuda.Recycle(dy12)
	Time += Dt_si

  // Rewrite to calculate spep 2
	torqueFnAFLLBPRB(dy11,dy12,Hth1a,Hth2a,Hth1b,Hth2b)

	err1 := cuda.MaxVecDiff(dy1, dy11) * float64(dt1)
	err2 := cuda.MaxVecDiff(dy2, dy12) * float64(dt2)
	err:=-1.0
	if err2>err1 {err=err2} else {err=err1}

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y1, y1, dy11, dy1, 1, 0.5*dt1, -0.5*dt1) //****
		cuda.Madd3(y2, y2, dy12, dy2, 1, 0.5*dt2, -0.5*dt2) //****
		// M is not normalized in LLB
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./2.))
		setLastErr(err)
		setMaxTorque(dy1)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(y1, y1, dy1, 1, -dt1)  //****
		cuda.Madd2(y2, y2, dy2, 1, -dt2)  //****
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
	}
}

func (_ *HeunAFLLBPRB054401) Free() {}

// Torques for antiferro

// write torque to dst and increment NEvals
// Now everything here to use 2 lattices at the same time
func torqueFnAFLLBPRB(dst1,dst2 *data.Slice,hth1a,hth2a,hth1b,hth2b *data.Slice) {

	// Set Effective field

	if (!isolatedlattices) {
		UpdateM()
		SetDemagField(dst1)
		data.Copy(dst2, dst1)
	} else {   // Just for code debug
		data.Copy(M.Buffer(), M1.Buffer())
		*Msat=*Msat1
		SetDemagField(dst1)
		data.Copy(M.Buffer(), M2.Buffer())
		*Msat=*Msat2
		SetDemagField(dst2)
	}
	AddExchangeFieldAF(dst1,dst2)
	AddAnisotropyFieldAF(dst1,dst2)
	//AddAFMExchangeField(dst)  // AFM Exchange non adjacent layers
	B_ext.AddTo(dst1)
	B_ext.AddTo(dst2)
	
	AddCustomField(dst1)
	AddCustomField(dst2)

	cuda.Zero(hth1a)       
	cuda.Zero(hth2a)       
	cuda.Zero(hth1b)       
	cuda.Zero(hth2b)       
	if JHThermalnoise == true {
		B_therm.AddTo(hth1a)
		B_therm.AddTo(hth2a)
		B_therm.AddTo(hth1b)
		B_therm.AddTo(hth2b)
	}
	
	// STT
	AddSTTorqueAF(dst1,dst2)
	
	// Add to sublattice 1 and 2
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	alpha1 := Alpha1.MSlice()
	defer alpha1.Recycle()
	alpha2 := Alpha2.MSlice()
	defer alpha2.Recycle()

	Tcurie := TCurie.MSlice()
	defer Tcurie.Recycle()
	Msat := Msat.MSlice()
	defer Msat.Recycle()
	Msat1 := Msat1.MSlice()
	defer Msat1.Recycle()
	Msat2 := Msat2.MSlice()
	defer Msat2.Recycle()
	temp := Temp.MSlice()
	defer temp.Recycle()

	X_TM := x_TM.MSlice()
	defer X_TM.Recycle()
	NV := nv.MSlice()
	defer NV.Recycle()

	MU1 := mu1.MSlice()
	defer MU1.Recycle()
	MU2 := mu2.MSlice()
	defer MU2.Recycle()

	J0AA := J0aa.MSlice()
	defer J0AA.Recycle()
	J0BB := J0bb.MSlice()
	defer J0BB.Recycle()
	J0AB := J0ab.MSlice()
	defer J0AB.Recycle()

	if Precess {
		cuda.LLBTorqueAFPRB(dst1, M1.Buffer(), dst2, M2.Buffer(), dst1,dst2, temp,alpha,alpha1,alpha2,Tcurie,Msat,Msat1,Msat2,hth1a,hth2a,hth1b,hth2b,X_TM,NV,MU1,MU2,J0AA,J0BB,J0AB) // overwrite dst with torque
	}  else {
		cuda.LLNoPrecess(dst1, M1.Buffer(), dst1)
		cuda.LLNoPrecess(dst2, M2.Buffer(), dst2)
	}

	FreezeSpins(dst1)
	FreezeSpins(dst2)

	NEvals++
}


