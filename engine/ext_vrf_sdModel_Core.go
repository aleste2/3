package engine

// sd Model core definitions anfd functions

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	rhosd    = NewScalarParam("rhosd", "J", "Scaling dm/dt to spin accumulation")
	tausf    = NewScalarParam("tausf", "s", "Relaxation time for mu")
	tausd    = NewScalarParam("tausd", "s", "Relaxation time for m-mu")
	Dbar     = NewScalarParam("Dbar", "m2/s", "Difussion constant for spin accumulation")
	Sigmabar = NewScalarParam("Sigmabar", "m2/s", "Difussion constant for spin accumulation")
	mus      magnetization // spin accumulation (J)
	rhov     = NewScalarParam("rhov", "1/m3", "Density volumen")
	B_mus    = NewVectorField("B_mus", "T", "Spin field", AddSpinField)
	Jspin    = NewVectorField("Js", "A/m2", "Spin current", AddJspin)
)

func init() {
	DeclLValue("mus", &mus, `Spin Acummulation (J)`)
	DeclFunc("InitsdModel", InitsdModel, "Init sd Model")
}

func InitsdModel() {
	mus.alloc()
	mus.name = "mu_s"
	cuda.Zero(mus.Buffer())
}

func musStep(dy11, dy1 *data.Slice) {

	// Add to sublattice 1 and 2
	Rhosd := rhosd.MSlice()
	defer Rhosd.Recycle()
	Tausf := tausf.MSlice()
	defer Tausf.Recycle()
	dbar := Dbar.MSlice()
	defer dbar.Recycle()
	sigmabar := Sigmabar.MSlice()
	defer sigmabar.Recycle()

	if mus.Buffer() == nil {
		mus.alloc()
		cuda.Zero(mus.Buffer())
	}

	// Buffers for sd model
	mus1 := mus.Buffer()
	dmus1 := cuda.Buffer(VECTOR, mus1.Size())
	defer cuda.Recycle(dmus1)

	if FixDt != 0 {
		Dt_si = FixDt
	}
	dt := float32(Dt_si * GammaLL)

	// Newton step

	/*
	   // Forced  Substeps
	           Substeps:=2000
	   		time0 := Time
	   		for iSubsteps := 0; iSubsteps < Substeps; iSubsteps++ {
	   			cuda.MusStep(dmus1, dt, M.Buffer(), mus1, Rhosd, Tausf, dbar, sigmabar, dy11, dy1, M.Mesh(), geometry.Gpu(), regions.Gpu()) // overwrite dst with torque
	   			cuda.Madd2(mus1, mus1, dmus1, 1, float32(Dt_si)/float32(Substeps))                                                          // y = y + dt * dy
	   			Time += float64(float32(Dt_si) / float32(Substeps))
	   			//print("Substes: ", iSubsteps, "\n")
	   		}
	   		Time = time0




	   //
	*/

	cuda.MusStep(dmus1, dt, M.Buffer(), mus1, Rhosd, Tausf, dbar, sigmabar, dy11, dy1, M.Mesh(), geometry.Gpu(), regions.Gpu()) // overwrite dst with torque
	err := cuda.MaxVecNorm(dmus1)
	tolerancemu := 1.0e-24
	if err*Dt_si < tolerancemu { //Good Step
		//print(err*Dt_si, "\n")
		cuda.Madd2(mus1, mus1, dmus1, 1, float32(Dt_si)) // y = y + dt * dy
	} else { // Substeps
		//print("Entro\n")
		time0 := Time
		//print("Substeps\n")
		Substeps := int(math.Ceil(float64(float32(err*Dt_si) / float32(tolerancemu))))
		if Substeps > 254 {
			Substeps = 255
		}
		//print("Substeps: ", Substeps, "\n")
		for iSubsteps := 0; iSubsteps < Substeps; iSubsteps++ {
			cuda.MusStep(dmus1, dt, M.Buffer(), mus1, Rhosd, Tausf, dbar, sigmabar, dy11, dy1, M.Mesh(), geometry.Gpu(), regions.Gpu()) // overwrite dst with torque
			cuda.Madd2(mus1, mus1, dmus1, 1, float32(Dt_si)/float32(Substeps))                                                          // y = y + dt * dy
			Time += float64(float32(Dt_si) / float32(Substeps))
			//print("Substes: ", iSubsteps, "\n")
		}
		Time = time0
		//print("Salgo\n")

	}

}

func TTMplus(dy11, dy1, LLBbufferTe, LLBbufferTeBig *data.Slice, dt float32) {
	Tc := TCurie.MSlice()
	defer Tc.Recycle()
	Rho := rhov.MSlice()
	defer Rho.Recycle()
	Cel := Ce.MSlice()
	defer Cel.Recycle()

	cuda.TTMplus(dy11, dy1, LLBbufferTe, LLBbufferTeBig, M.Buffer(), Tc, Rho, Cel, dt)
}

func MusMoment(dt float32) {
	Tc := TCurie.MSlice()
	defer Tc.Recycle()
	Tauds := tausd.MSlice()
	defer Tauds.Recycle()
	cuda.AddMagneticmoment(M.Buffer(), mus.Buffer(), Tc, Tauds, dt)
}

func AddMusEffectiveField(dst *data.Slice) {
	Tc := TCurie.MSlice()
	defer Tc.Recycle()
	Ms := Msat.MSlice()
	defer Ms.Recycle()
	cuda.MusEffectiveField(dst, mus.Buffer(), M.Buffer(), Tc, Ms)
}

func AddSpinField(dst *data.Slice) {
	AddMusEffectiveField(dst)
}

func AddJspin(dst *data.Slice) {
	sbar := Sigmabar.MSlice()
	defer sbar.Recycle()
	cuda.AddJspin(dst, mus.Buffer(), sbar, M.Mesh(), geometry.Gpu())
}
