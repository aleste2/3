package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// For LLB the same functions as for LLG

// Sets dst to the current total LLBtorque
func SetTorqueLLB(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {
	SetLLBTorque(dst, hth1, hth2)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz-Bloch torque
func SetLLBTorque(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {

	SetEffectiveField(dst) // calc and store B_eff
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	TCurie := TCurie.MSlice()
	defer TCurie.Recycle()
	Msat := Msat.MSlice()
	defer Msat.Recycle()
	Temp := Temp.MSlice()
	defer Temp.Recycle()

	A1 := a1.MSlice()
	defer A1.Recycle()

	//cuda.Zero(hth1)
	//if (JHThermalnoise==true) {B_therm.AddTo(hth1)}
	//cuda.Zero(hth2)
	//if (JHThermalnoise==true) {B_therm.AddTo(hth2)}
	if Precess {
		if MFA == false {
			cuda.LLBTorque(dst, M.Buffer(), dst, Temp, alpha, TCurie, Msat, hth1, hth2, Langevin, A1) // overwrite dst with torque
		} else {
			NV := nv.MSlice()
			defer NV.Recycle()
			MU1 := mu1.MSlice()
			defer MU1.Recycle()
			J0AA := J0aa.MSlice()
			defer J0AA.Recycle()
			cuda.LLBTorqueMFA(dst, M.Buffer(), dst, Temp, alpha, Msat, hth1, hth2, NV, MU1, J0AA)
		}
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

///////////////// For LLBJH the same previous two functions

// Sets dst to the current total LLBtorque
func SetTorqueLLBJH(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {
	SetLLBTorqueJH(dst, hth1, hth2)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz-Bloch torque
func SetLLBTorqueJH(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {
	SetEffectiveField(dst) // calc and store B_eff
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	TCurie := TCurie.MSlice()
	defer TCurie.Recycle()
	Msat := Msat.MSlice()
	defer Msat.Recycle()

	A1 := a1.MSlice()
	defer A1.Recycle()

	//cuda.Zero(hth1)
	//B_therm.LLBAddTo(hth1)
	//cuda.Zero(hth2)
	//B_therm.LLBAddTo(hth2)
	if Precess {
		cuda.LLBTorqueJH(dst, M.Buffer(), dst, TempJH.temp, alpha, TCurie, Msat, hth1, hth2, Langevin, A1) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

///////////////// For 3T model the same previous two functions

// Sets dst to the current total LLBtorque
func SetTorqueLLB3T(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {
	SetLLBTorque3T(dst, hth1, hth2)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz-Bloch torque
func SetLLBTorque3T(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {
	SetEffectiveField(dst) // calc and store B_eff
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	TCurie := TCurie.MSlice()
	defer TCurie.Recycle()
	Msat := Msat.MSlice()
	defer Msat.Recycle()

	A1 := a1.MSlice()
	defer A1.Recycle()

	//cuda.Zero(hth1)
	//B_therm.LLBAddTo(hth1)
	//cuda.Zero(hth2)
	//B_therm.LLBAddTo(hth2)
	if Precess {
		cuda.LLBTorqueJH(dst, M.Buffer(), dst, Ts.temp, alpha, TCurie, Msat, hth1, hth2, Langevin, A1) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}

///////////////// For 2T model the same previous two functions

// Sets dst to the current total LLBtorque
func SetTorqueLLB2T(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {
	SetLLBTorque2T(dst, hth1, hth2)
	AddSTTorque(dst)
	FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz-Bloch torque
func SetLLBTorque2T(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {
	SetEffectiveField(dst) // calc and store B_eff
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	TCurie := TCurie.MSlice()
	defer TCurie.Recycle()
	Msat := Msat.MSlice()
	defer Msat.Recycle()

	A1 := a1.MSlice()
	defer A1.Recycle()

	//cuda.Zero(hth1)
	//B_therm.LLBAddTo(hth1)
	//cuda.Zero(hth2)
	//B_therm.LLBAddTo(hth2)
	if Precess {
		cuda.LLBTorqueJH(dst, M.Buffer(), dst, Te.temp, alpha, TCurie, Msat, hth1, hth2, Langevin, A1) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}
}
