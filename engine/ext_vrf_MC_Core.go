package engine

// MC Core

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	generator     curand.Generator
	NumberMCSteps int
	Ncell         = -1e0
	delta         = 0.1
)

func init() {
	DeclFunc("MCsteps", MCsteps, "Monte Carlo Minimizer")
}

// New Stepper like
type MCstep struct {
	McSteps int64
}

func (MCstep *MCstep) Step() {

	// old M to restore bad chages
	mold := cuda.Buffer(VECTOR, M.Buffer().Size())
	defer cuda.Recycle(mold)

	BeffMC := cuda.Buffer(VECTOR, M.Buffer().Size())
	defer cuda.Recycle(BeffMC)

	// Energy density buffers
	uold := cuda.Buffer(1, M.Buffer().Size())
	defer cuda.Recycle(uold)
	unew := cuda.Buffer(1, M.Buffer().Size())
	defer cuda.Recycle(unew)

	// Rates
	changed := cuda.Buffer(1, M.Buffer().Size())
	defer cuda.Recycle(changed)
	cuda.Zero(changed)

	// Random numbers
	dmx := cuda.Buffer(1, M.Buffer().Size())
	defer cuda.Recycle(dmx)
	dmy := cuda.Buffer(1, M.Buffer().Size())
	defer cuda.Recycle(dmy)
	dmz := cuda.Buffer(1, M.Buffer().Size())
	defer cuda.Recycle(dmz)
	du := cuda.Buffer(1, M.Buffer().Size())
	defer cuda.Recycle(du)
	// for LLB
	dmm := cuda.Buffer(1, M.Buffer().Size())
	defer cuda.Recycle(dmm)

	if generator == 0 {
		generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
	}
	if Ncell == -1 {
		dst := cuda.Buffer(1, M.Buffer().Size())
		defer cuda.Recycle(dst)
		cuda.Zero(dst)
		data.Copy(mold, M.Buffer())
		M.normalize()
		cuda.AddDotProduct(dst, 1, M.Buffer(), M.Buffer())
		data.Copy(M.Buffer(), mold)
		Ncell = float64(cuda.Sum(dst))
		//print(Ncell)
	}

	N := Mesh().NCell()

	generator.GenerateNormal(uintptr(dmx.DevPtr(0)), int64(N), 0, 1)
	generator.GenerateNormal(uintptr(dmy.DevPtr(0)), int64(N), 0, 1)
	generator.GenerateNormal(uintptr(dmz.DevPtr(0)), int64(N), 0, 1)
	generator.GenerateUniform(uintptr(du.DevPtr(0)), int64(N))
	if LLBeq == true {
		generator.GenerateUniform(uintptr(dmm.DevPtr(0)), int64(N))
		MCCalcdm(dmm)
	}

	if EnableDemag == true {
		SetDemagField(BeffMC)
	}

	norm0 := 0.0
	if LLBeq == false {
		norm0 = 1
	}

	for net := 0; net < 2; net++ {

		//delta=0.0
		/*
			generator.GenerateNormal(uintptr(dmx.DevPtr(0)), int64(N), 0, 1)
			generator.GenerateNormal(uintptr(dmy.DevPtr(0)), int64(N), 0, 1)
			generator.GenerateNormal(uintptr(dmz.DevPtr(0)), int64(N), 0, 1)
			generator.GenerateUniform(uintptr(du.DevPtr(0)), int64(N))
			generator.GenerateUniform(uintptr(dmm.DevPtr(0)), int64(N))
		*/
		MCGetU(uold, BeffMC, mold)
		data.Copy(mold, M.Buffer())
		//if LLBeq == true {
		//	MCCalcdm(dmm)
		//}
		MCmove(M.Buffer(), dmx, dmy, dmz, dmm, net, float32(delta), float32(norm0))
		MCGetU(unew, BeffMC, mold)
		MCCompare(unew, uold, M.Buffer(), mold, du, net, changed)
	}

	if NumberMCSteps%50 == 0 {
		//err := 100 * float64(cuda.Sum(changed)/float32(N))
		ChangedPercent := 100 * float64(cuda.Sum(changed)/float32(Ncell))
		if ChangedPercent < 40 {
			delta = delta / 1.23 //0.73 // / 2
			if delta < 0.00001 {
				delta = 0.00001
				//if delta > 1000 {
				//	delta = 1000
			}
		}
		if ChangedPercent > 50 {
			delta = delta * 1.27 // * 2
			if delta > 10000 {
				delta = 10000
				//if delta < 0.0001 {
				//	delta = 0.0001
			}
		}
		MaxDt = delta
		setLastErr(ChangedPercent)
	}
	NumberMCSteps++
	NSteps++
}

func (MCstep *MCstep) Free() {
}

// New MCSteps

func MCsteps(steps int) {

	//if (LLBeq == true) || (AFf == true) {
	if AFf == true {
		util.AssertMsg(0 > 1, "LLB and AF still not supported in MC")
	}

	// Save the settings we are changing...
	prevType := solvertype
	prevFixDt := FixDt
	prevPrecess := Precess
	t0 := Time
	relaxing = true // disable temperature noise
	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		FixDt = prevFixDt
		Precess = prevPrecess
		Time = t0
		relaxing = false
	}()

	Precess = false // disable precession for torque calculation
	// remove previous stepper
	if stepper != nil {
		stepper.Free()
	}

	stepper = new(MCstep)

	NumberMCSteps = 0
	cond := func() bool {
		return (NumberMCSteps <= steps)
	}

	oldMaxDt := MaxDt
	RunWhile(cond)
	pause = true
	stepper.Free()
	Ncell = -1
	MaxDt = oldMaxDt

}

func MCGetU(dst, BeffMC, mold *data.Slice) {

	// LLG no AF
	//if (LLBeq == false) && (AFf == false) && (Ato == false) {
	if (AFf == false) && (Ato == false) {
		cuda.Zero(dst)
		if LLBeq == false {
			AddExchangeEnergyDensity(dst)
			cuda.Madd2(dst, dst, dst, 1, 1) // ×2 solo para la densidad estándar (−½)
		} else {
			AddLLBExchangeEnergyDensity(dst) // ya es la energía de sitio completa
		}
		AddAnisotropyEnergyDensity(dst)
		AddMagnetoelasticEnergyDensity(dst)
		AddCustomEnergyDensity(dst)
		AddEdens_zeeman(dst)

		if EnableDemag == true {
			mf := ValueOf(M_full)
			defer cuda.Recycle(mf)
			//factor := float32(-0.5)
			factor := float32(-1.0)
			cuda.AddDotProduct(dst, factor, BeffMC, mf)
		}

		if LLBeq == true {
			//cuda.Zero(dst)
			AddLLBEnergyDensity(dst)
		}
	}

	// Atomistic
	if Ato == true {

		// Volume
		c := M.Mesh().CellSize()
		invVol := float32(1 / (c[X] * c[Y] * c[Z]))

		mfull := cuda.Buffer(3, M.Buffer().Size())
		defer cuda.Recycle(mfull)
		SetMFullAto(mfull)
		beff := cuda.Buffer(3, M.Buffer().Size())
		defer cuda.Recycle(beff)
		cuda.Zero(dst)
		cuda.Zero(beff)
		AddExchangeFieldAto(beff)
		//B_ext.AddTo(beff)
		//cuda.AddDotProduct(dst, -1*invVol, beff, mfull) // factor 1 to avoid *2
		cuda.AddDotProduct(dst, -1.*invVol, beff, mfull) // factor 1 to avoid *2
		cuda.Zero(beff)
		AddAnisotropyFieldAto(beff, M, mu, Dato, AnisU)
		cuda.AddDotProduct(dst, -1./2.*invVol, beff, mfull)
		cuda.Zero(beff)
		B_ext.AddTo(beff)
		cuda.AddDotProduct(dst, -1.*invVol, beff, mfull)
	}
}

func MCmove(mnew, x, y, z, dmm *data.Slice, iteration int, delta, norm float32) {
	cuda.MCmove(mnew, x, y, z, dmm, iteration, delta, norm, M.Mesh())
}

func MCCompare(unew, uold, mnew, mold, rnd *data.Slice, iteration int, changed *data.Slice) {
	temp := Temp.MSlice()
	defer temp.Recycle()
	cuda.MCCompare(unew, uold, mnew, mold, rnd, temp, iteration, M.Mesh(), changed, float32(Ncell))
}

func AddLLBEnergyDensity(dst *data.Slice) {
	Tcurie := TCurie.MSlice()
	defer Tcurie.Recycle()
	Ms := Msat.MSlice()
	defer Ms.Recycle()
	temp := Temp.MSlice()
	defer temp.Recycle()
	Mua := mu1.MSlice()
	defer Mua.Recycle()
	cuda.AddLLBEnergyDensity(dst, M.Buffer(), Ms, temp, Tcurie, Mua)
}

func MCCalcdm(dst *data.Slice) {
	Tcurie := TCurie.MSlice()
	defer Tcurie.Recycle()
	Ms := Msat.MSlice()
	defer Ms.Recycle()
	temp := Temp.MSlice()
	defer temp.Recycle()
	Mua := mu1.MSlice()
	defer Mua.Recycle()
	// Volume
	c := M.Mesh().CellSize()
	vol := float32((c[X] * c[Y] * c[Z]))
	cuda.MCCalcdm(dst, M.Buffer(), Ms, temp, Tcurie, Mua, vol)
}

func AddLLBExchangeEnergyDensity(dst *data.Slice) {
	ms := Msat.MSlice()
	defer ms.Recycle()
	cuda.AddLLBExchangeEnergyDensity(dst, M.Buffer(), lex2.Gpu(), ms, regions.Gpu(), M.Mesh())
}
