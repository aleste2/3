package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
	"math"
	"math/rand"
)

var (
	Jato = NewScalarParam("Jato", "J", "AtomisticsExchange", &Jexato)
	Jdmi = NewScalarParam("Jdmi", "J", "AtomisticsDMIExchange", &Jdmiato)
	Dato = NewScalarParam("Dato", "J", "Atomistic uniaxial anisotropy constant")
	mu   = NewScalarParam("mu", "J/T", "Atomistic magnetic moment")
	g    = NewScalarParam("g", "a.u.", "Lande Factor")

	Jexato  exchParam // inter-cell exchange
	Jdmiato exchParam // inter-cell exchange

	B_ato      = NewVectorField("B_ato", "T", "Effective Atomistic field", SetEffectiveFieldAto)
	M_full_ato = NewVectorField("m_full_ato", "A/m", "Unnormalized magnetization", SetMFullato)
	celltype   = 0
)

// Sets the exchange interaction between region 1 and 2.
func InterAtoExchange(region1, region2 int, value float64) {
	Jexato.setInter(region1, region2, value)
}
func InterAtoDMI(region1, region2 int, value float64) {
	Jdmiato.setInter(region1, region2, value)
}

func init() {
	Jexato.init(Jato)
	Jdmiato.init(Jdmi)
	DeclFunc("BCC", bcc_init, "Initialize BCC lattice")
	DeclFunc("SC", sc_init, "Initialize SC lattice")
	DeclFunc("sc_AF", sc_AF, "Initialize AF in checkboard")
	DeclFunc("FCC", fcc_init, "Initialize FCC lattice")
	DeclFunc("alloy", alloy, "Set alloy percent")
	DeclFunc("ext_InterAtoExchange", InterAtoExchange, "Sets exchange coupling between two regions.")
	DeclFunc("ext_InterAtoDMI", InterAtoDMI, "Sets DMI coupling between two regions.")
	DeclFunc("ext_GetAtoEnergy", GetAtoEnergy, "Gets energy in atomistic calculations.")
	DeclFunc("ext_GetAtoEnergy01", GetAtoEnergy01, "Gets 0 1 energy in atomistic calculations.")
}

func SetMFullato(dst *data.Slice) {
	// scale m by Msat...
	mu, rM := mu.Slice()
	if rM {
		defer cuda.Recycle(mu)
	}
	for c := 0; c < 3; c++ {
		cuda.Mul(dst.Comp(c), M.Buffer().Comp(c), mu)
	}
}

func GetAtoEnergy() float64 {
	return -1.0 * dot(M_full_ato, B_ato)
}

func GetAtoEnergy01(r int) float64 {
	oldBeff := B_ato

	// Backup mu values
	mu00 := mu.getRegion(0)
	mu10 := mu.getRegion(1)
	zero := []float64{0.0}

	if r == 0 {
		mu.setRegion(1, zero)
	}
	if r == 1 {
		mu.setRegion(0, zero)
	}
	//en1:=-1.0* dot(M_full_ato,oldBeff)
	en1 := -1.0 * dot(M_full_ato, oldBeff)
	mu.setRegion(0, mu00)
	mu.setRegion(1, mu10)

	return en1
}

func alloyold(host, alloy int, percent float64) {
	count := 0.0
	n := M.Mesh().Size()

	for i := 0; i < n[X]; i++ {
		for j := 0; j < n[Y]; j++ {
			for k := 0; k < n[Z]; k++ {
				//dp:=DotProduct(M.GetCell(i,j,k),M.GetCell(i,j,k))
				if (M.GetCell(i, j, k)[0]*M.GetCell(i, j, k)[0]+M.GetCell(i, j, k)[1]*M.GetCell(i, j, k)[1]+M.GetCell(i, j, k)[2]*M.GetCell(i, j, k)[2] != 0) && (regions.GetCell(i, j, k) == host) {
					count++
				}
			}
		}
	}
	print(count, " host atoms\n")
	added := 0.0
	for added < percent*count {
		i := rand.Intn(n[X])
		j := rand.Intn(n[Y])
		k := rand.Intn(n[Z])
		if (M.GetCell(i, j, k)[0]*M.GetCell(i, j, k)[0]+M.GetCell(i, j, k)[1]*M.GetCell(i, j, k)[1]+M.GetCell(i, j, k)[2]*M.GetCell(i, j, k)[2] != 0) && (regions.GetCell(i, j, k) == host) {
			regions.SetCell(i, j, k, alloy)
			added++
		}
	}
	print(added, " alloy atoms\n")
}

type alloyst struct {
	seed      int64            // seed for generator
	generator curand.Generator //
	noise     *data.Slice      // noise buffer
}

func alloy(host, alloy int, percent float64) {
	var ralloy alloyst
	ralloy.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
	ralloy.seed = 1
	y := M.Buffer()
	ralloy.noise = cuda.Buffer(1, y.Size())
	defer cuda.Recycle(ralloy.noise)
	N := Mesh().NCell()
	ralloy.generator.GenerateUniform(uintptr(ralloy.noise.DevPtr(0)), int64(N))
	cuda.Alloypar(host, alloy, percent, ralloy.noise, regions.gpuCache.Ptr)
}

func sc_init() {
	nv.Set(6)
	celltype = 0
	print("Cellsize must be given in unit cells\n")
}

func sc_AF(alloy int) {
	ii := 0
	jj := 0
	kk := 0
	n := M.Mesh().Size()
	nv.Set(6)
	for i := 0; i < n[X]/2+1; i += 1 {
		for j := 0; j < n[Y]/2+1; j += 1 {
			for k := 0; k < n[Z]/2+1; k += 1 {
				ii = i*2 + 1
				jj = j * 2
				kk = k * 2
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, alloy)
				}
				ii = i * 2
				jj = j*2 + 1
				kk = k * 2
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, alloy)
				}
				ii = i * 2
				jj = j * 2
				kk = k*2 + 1
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, alloy)
				}
				ii = i*2 + 1
				jj = j*2 + 1
				kk = k*2 + 1
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, alloy)
				}
			}
		}
	}
}

func bcc_init() {
	print("Cellsize must be given in half of unit cells\n")
	celltype = 1
	ii := 0
	jj := 0
	kk := 0
	n := M.Mesh().Size()
	nv.Set(27)
	print("Region 255 used for empty atomic positions\n")
	for i := 0; i < n[X]/2+1; i += 1 {
		for j := 0; j < n[Y]/2+1; j += 1 {
			for k := 0; k < n[Z]/2+1; k += 1 {
				ii = i*2 + 1
				jj = j * 2
				kk = k * 2
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, 255)
				}
				ii = i * 2
				jj = j*2 + 1
				kk = k * 2
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, 255)
				}
				ii = i*2 + 1
				jj = j*2 + 1
				kk = k * 2
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, 255)
				}
				ii = i * 2
				jj = j * 2
				kk = k*2 + 1
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, 255)
				}
				ii = i*2 + 1
				jj = j * 2
				kk = k*2 + 1
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, 255)
				}
				ii = i * 2
				jj = j*2 + 1
				kk = k*2 + 1
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, 255)
				}
			}
		}
	}
	M.SetRegion(255, Uniform(0, 0, 0))
}

func fcc_init() {
	print("Cellsize must be given in half of unit cells\n")
	celltype = 2
	ii := 0
	jj := 0
	kk := 0
	n := M.Mesh().Size()
	nv.Set(27)
	print("Region 255 used for empty atomic positions\n")
	//SetGeom(Xrange(-Inf,Inf))
	//SetGeom(Rect(1,1))
	for i := 0; i < n[X]/2+1; i += 1 {
		for j := 0; j < n[Y]/2+1; j += 1 {
			for k := 0; k < n[Z]/2+1; k += 1 {
				ii = i*2 + 1
				jj = j * 2
				kk = k * 2
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, 255)
					//cuda.SetCell(geometry.buffer, 0, ii, jj, kk, 0) // a bit slowish, but hardly reached
				}
				ii = i * 2
				jj = j*2 + 1
				kk = k * 2
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, 255)
					//cuda.SetCell(geometry.buffer, 0, ii, jj, kk, 0) // a bit slowish, but hardly reached
				}
				ii = i * 2
				jj = j * 2
				kk = k*2 + 1
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, 255)
					//cuda.SetCell(geometry.buffer, 0, ii, jj, kk, 0) // a bit slowish, but hardly reached
				}
				ii = i*2 + 1
				jj = j*2 + 1
				kk = k*2 + 1
				if (ii < n[X]) && (jj < n[Y]) && (kk < n[Z]) {
					regions.SetCell(ii, jj, kk, 255)
					//cuda.SetCell(geometry.buffer, 0, ii, jj, kk, 0) // a bit slowish, but hardly reached
				}
			}
		}
	}
	M.SetRegion(255, Uniform(0, 0, 0))
}

// Adaptive Heun solver.
type HeunAto struct{}

// Adaptive Heun method, can be used as solver.Step
func (_ *HeunAto) Step() {
	y := M.Buffer()
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si * GammaLL)
	util.Assert(dt > 0)

	// Temperature giving problems
	if (!Temp.isZero()) || (LLB2Tf == true) {
		B_therm.updateAto()
	}

	// stage 1
	torqueAto(dy0)
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy

	// stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)
	Time += Dt_si
	torqueAto(dy)

	err := cuda.MaxVecDiff(dy0, dy) * float64(dt)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
		M.normalize()
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./2.))
		setLastErr(err)
		setMaxTorque(dy)
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(y, y, dy0, 1, -dt)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
		//print("No debo entrar aqui\n")
	}
}

func (_ *HeunAto) Free() {}

type HeunAto2T struct{}

// Adaptive Heun method for 2T model, can be used as solver.Step
func (_ *HeunAto2T) Step() {
	y := M.Buffer()
	dy0 := cuda.Buffer(VECTOR, y.Size())
	defer cuda.Recycle(dy0)

	if FixDt != 0 {
		Dt_si = FixDt
	}

	dt := float32(Dt_si * GammaLL)
	util.Assert(dt > 0)

	// Temperature giving problems
	if (!Temp.isZero()) || (LLB2Tf == true) {
		B_therm.updateAto()
	}

	// stage 1
	torqueAto(dy0)
	cuda.Madd2(y, y, dy0, 1, dt) // y = y + dt * dy

	// stage 2
	dy := cuda.Buffer(3, y.Size())
	defer cuda.Recycle(dy)
	Time += Dt_si
	torqueAto(dy)

	err := cuda.MaxVecDiff(dy0, dy) * float64(dt)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		cuda.Madd3(y, y, dy, dy0, 1, 0.5*dt, -0.5*dt)
		M.normalize()
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./2.))
		setLastErr(err)
		setMaxTorque(dy)
		for iter := 0; iter < TSubsteps; iter++ {
			NewtonStep2T(float32(Dt_si) / float32(TSubsteps))
		}
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time -= Dt_si
		cuda.Madd2(y, y, dy0, 1, -dt)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./3.))
		//		print("No debo entrar aqui\n")
	}
}

func (_ *HeunAto2T) Free() {}

// write torque to dst and increment NEvals
func torqueAto(dst *data.Slice) {
	SetTorqueAto(dst)
	NEvals++
}

// Sets dst to the current total torque
func SetTorqueAto(dst *data.Slice) {
	SetLLTorqueAto(dst)
	//AddSTTorque(dst)
	//FreezeSpins(dst)
}

// Sets dst to the current Landau-Lifshitz torque
func SetLLTorqueAto(dst *data.Slice) {
	SetEffectiveFieldAto(dst) // calc and store B_eff

	alpha := Alpha.MSlice()
	defer alpha.Recycle()

	//cuda.LLTorque(dst, M.Buffer(), dst, alpha) // overwrite dst with torque
	if Precess {
		cuda.LLTorque(dst, M.Buffer(), dst, alpha) // overwrite dst with torque
	} else {
		cuda.LLNoPrecess(dst, M.Buffer(), dst)
	}

	AddSTTorqueAto(dst)
	multiplyLandeFactor(dst)
	/// Remember to add lande factor and multiply
}

func SetEffectiveFieldAto(dst *data.Slice) {
	cuda.Zero(dst)
	//	AddSTTorqueAto(dst)
	//	SetDemagField(dst)    // set to B_demag...
	B_therm.AddToAto(dst)
	AddExchangeFieldAto(dst) // ...then add other terms
	AddAnisotropyFieldAto(dst, M, mu, Dato, AnisU)
	B_ext.AddTo(dst)
}

// Adds the current exchange field to dst
func AddExchangeFieldAto(dst *data.Slice) {
	Mu := mu.MSlice()
	defer Mu.Recycle()
	Nv := nv.MSlice()
	defer Nv.Recycle()
	cuda.AddExchangeAto(dst, M.Buffer(), Jexato.Gpu(), Jdmiato.Gpu(), Mu, Nv, regions.Gpu(), M.Mesh())
}

// Adds the current uniaxial anisotropy
func AddAnisotropyFieldAto(dst *data.Slice, M magnetization, mu, Dato *RegionwiseScalar, AnisU *RegionwiseVector) {
	if Dato.nonZero() {
		Mu := mu.MSlice()
		defer Mu.Recycle()
		dato := Dato.MSlice()
		defer dato.Recycle()
		u := AnisU.MSlice()
		defer u.Recycle()

		cuda.AddUniaxialAnisotropyAto(dst, M.Buffer(), Mu, dato, u)
	}
}

// Adds the current spin transfer torque to dst
func AddSTTorqueAto(dst *data.Slice) {
	if !J.isZero() {
		util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
		jspin, rec := J.Slice()
		if rec {
			defer cuda.Recycle(jspin)
		}
		fl, rec := FixedLayer.Slice()
		if rec {
			defer cuda.Recycle(fl)
		}
		if !DisableZhangLiTorque {
			msat := mu.MSlice()
			//msat := Msat.MSlice()
			defer msat.Recycle()
			j := J.MSlice()
			defer j.Recycle()
			alpha := Alpha.MSlice()
			defer alpha.Recycle()
			xi := Xi.MSlice()
			defer xi.Recycle()
			pol := Pol.MSlice()
			defer pol.Recycle()
			cuda.AddZhangLiTorqueAto(dst, M.Buffer(), msat, j, alpha, xi, pol, Mesh(),celltype)
			//multiplyVolume(dst) // Rewrite STT in cuda!!!!
		}
		if !DisableSlonczewskiTorque && !FixedLayer.isZero() {
			msat := mu.MSlice()
			//		msat := Msat.MSlice()
			defer msat.Recycle()
			j := J.MSlice()
			defer j.Recycle()
			fixedP := FixedLayer.MSlice()
			defer fixedP.Recycle()
			alpha := Alpha.MSlice()
			defer alpha.Recycle()
			pol := Pol.MSlice()
			defer pol.Recycle()
			lambda := Lambda.MSlice()
			defer lambda.Recycle()
			epsPrime := EpsilonPrime.MSlice()
			defer epsPrime.Recycle()
			thickness := FreeLayerThickness.MSlice()
			defer thickness.Recycle()
			cuda.AddSlonczewskiTorque2Ato(dst, M.Buffer(),
				msat, j, fixedP, alpha, pol, lambda, epsPrime,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh(), celltype)
		}
	}
}

func multiplyLandeFactor(dst *data.Slice) {
	lande := g.MSlice()
	defer lande.Recycle()
	cuda.MultiplyLandeFactor(dst, lande)
}

func multiplyVolume(dst *data.Slice) {
	cuda.MultiplyVolume(dst, Mesh(), celltype)
}

func (b *thermField) AddToAto(dst *data.Slice) {
	if (!Temp.isZero()) || (LLB2Tf == true) {
		//b.updateAto()
		cuda.Add(dst, dst, b.noise)
	}
}

func (b *thermField) updateAto() {
	// we need to fix the time step here because solver will not yet have done it before the first step.
	// FixDt as an lvalue that sets Dt_si on change might be cleaner.
	if FixDt != 0 {
		Dt_si = FixDt
	}

	if b.generator == 0 {
		b.generator = curand.CreateGenerator(curand.PSEUDO_DEFAULT)
		b.generator.SetSeed(b.seed)
	}
	if b.noise == nil {
		b.noise = cuda.NewSlice(b.NComp(), b.Mesh().Size())
		// when noise was (re-)allocated it's invalid for sure.
		B_therm.step = -1
		B_therm.dt = -1
	}

	if Temp.isZero() {
		cuda.Memset(b.noise, 0, 0, 0)
		b.step = NSteps
		b.dt = Dt_si
		if !LLB2Tf == true {
			return
		}
	}
	//		print("2T-0\n")
	// keep constant during time step
	if NSteps == b.step && Dt_si == b.dt {
		if !LLB2Tf == true {
			return
		}
	}

	// after a bad step the timestep is rescaled and the noise should be rescaled accordingly, instead of redrawing the random numbers
	if NSteps == b.step && Dt_si != b.dt {
		for c := 0; c < 3; c++ {
			cuda.Madd2(b.noise.Comp(c), b.noise.Comp(c), b.noise.Comp(c), float32(math.Sqrt(b.dt/Dt_si)), 0.)
		}
		b.dt = Dt_si
		return
	}

	if FixDt == 0 {
		Refer("leliaert2017")
		//uncomment to not allow adaptive step
		//util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}
	//		print("2T-1\n")
	N := Mesh().NCell()
	k2_VgammaDt := 2 * mag.Kb / (GammaLL * Dt_si)
	noise := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(noise)

	const mean = 0
	const stddev = 1
	dst := b.noise
	Mu := mu.MSlice()
	defer Mu.Recycle()
	temp := Temp.MSlice()
	defer temp.Recycle()
	scaleNoise := ScaleNoiseLLB
	//scaleNoise := ScaleNoiseLLB.MSlice()
	//defer scaleNoise.Recycle()

	alpha := Alpha.MSlice()
	defer alpha.Recycle()

	for i := 0; i < 3; i++ {
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		if LLB2Tf == true {
			Te.update()
			//print("2T\n")
			cuda.SetTemperatureJH(dst.Comp(i), noise, k2_VgammaDt, Mu, Te.temp, alpha, scaleNoise)
		} else {
			cuda.SetTemperature(dst.Comp(i), noise, k2_VgammaDt, Mu, temp, alpha, scaleNoise)
		}
	}

	b.step = NSteps
	b.dt = Dt_si
}
