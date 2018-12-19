package engine

// Antiferromagnetic codes. Includes definitios adn changes to AF torques and thermal field

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/cuda/curand"
)

// Anisotropy, magnetization, exchange... variables and magenetization for sublattices
var (
	Ku11        = NewScalarParam("Ku11", "J/m3", "1st order uniaxial anisotropy constant")
	Ku21        = NewScalarParam("Ku21", "J/m3", "1st order uniaxial anisotropy constant")
	Ku12        = NewScalarParam("Ku12", "J/m3", "1st order uniaxial anisotropy constant")
	Ku22        = NewScalarParam("Ku22", "J/m3", "1st order uniaxial anisotropy constant")
	Kc11        = NewScalarParam("Kc11", "J/m3", "1st order cubic anisotropy constant")
	Kc21        = NewScalarParam("Kc21", "J/m3", "2nd order cubic anisotropy constant")
	Kc31        = NewScalarParam("Kc31", "J/m3", "3rd order cubic anisotropy constant")
	Kc12        = NewScalarParam("Kc12", "J/m3", "1st order cubic anisotropy constant")
	Kc22        = NewScalarParam("Kc22", "J/m3", "2nd order cubic anisotropy constant")
	Kc32        = NewScalarParam("Kc32", "J/m3", "3rd order cubic anisotropy constant")
	AnisU1      = NewVectorParam("anisU1", "", "Uniaxial anisotropy direction")
	AnisC11     = NewVectorParam("anisC11", "", "Cubic anisotropy direction #1")
	AnisC21     = NewVectorParam("anisC21", "", "Cubic anisotorpy directon #2")
	AnisU2      = NewVectorParam("anisU2", "", "Uniaxial anisotropy direction")
	AnisC12     = NewVectorParam("anisC12", "", "Cubic anisotropy direction #1")
	AnisC22     = NewVectorParam("anisC22", "", "Cubic anisotorpy directon #2")
	M1 magnetization // reduced magnetization (unit length)
	M2 magnetization // reduced magnetization (unit length)
	Msat1        = NewScalarParam("Msat1", "A/m", "Saturation magnetization", &lex21, &din21, &dbulk21)
	M_full1      = NewVectorField("m_full1", "A/m", "Unnormalized magnetization", SetMFull1)
	Msat2        = NewScalarParam("Msat2", "A/m", "Saturation magnetization", &lex22, &din22, &dbulk22)
	M_full2      = NewVectorField("m_full2", "A/m", "Unnormalized magnetization", SetMFull2)
	GammaLL1                 float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	GammaLL2                 float64 = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	Pol1          = NewScalarParam("Pol1", "", "Electrical current polarization Zhang-Li Lattice 1")
	Pol2          = NewScalarParam("Pol2", "", "Electrical current polarization Zhang-Li Lattice 2")
	isolatedlattices  bool = false // Debug only

)


func init() {
	DeclFunc("UpdateM", UpdateM, "UpdateM")
	DeclFunc("InitAntiferro", InitAntiferro, "InitAntiferro")
	DeclLValue("m1", &M1, `Reduced magnetization sublattice 1 (unit length)`)
	DeclLValue("m2", &M2, `Reduced magnetization sublattice 2 (unit length)`)
	M1.name="m1_"
	M2.name="m2_"
	DeclVar("GammaLL1", &GammaLL1, "Gyromagnetic ratio in rad/Ts Lattice 1")
	DeclVar("GammaLL2", &GammaLL2, "Gyromagnetic ratio in rad/Ts Lattice 2")
	DeclVar("isolatedlattices", &isolatedlattices, "Isolate AF lattices")
}

func InitAntiferro() {
	M1.alloc()
	M2.alloc()
}

// full magnetization for each subnet
func SetMFull1(dst *data.Slice) {
	// scale m by Msat...
	msat, rM := Msat1.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}
	for c := 0; c < 3; c++ {
		cuda.Mul(dst.Comp(c), M1.Buffer().Comp(c), msat)
	}

	// ...and by cell volume if applicable
	vol, rV := geometry.Slice()
	if rV {
		defer cuda.Recycle(vol)
	}
	if !vol.IsNil() {
		for c := 0; c < 3; c++ {
			cuda.Mul(dst.Comp(c), dst.Comp(c), vol)
		}
	}
}

func SetMFull2(dst *data.Slice) {
	// scale m by Msat...
	msat, rM := Msat2.Slice()
	if rM {
		defer cuda.Recycle(msat)
	}
	for c := 0; c < 3; c++ {
		cuda.Mul(dst.Comp(c), M2.Buffer().Comp(c), msat)
	}

	// ...and by cell volume if applicable
	vol, rV := geometry.Slice()
	if rV {
		defer cuda.Recycle(vol)
	}
	if !vol.IsNil() {
		for c := 0; c < 3; c++ {
			cuda.Mul(dst.Comp(c), dst.Comp(c), vol)
		}
	}
}

// Torques for antiferro

// write torque to dst and increment NEvals
// Now everything here to use 2 lattices at the same time
func torqueFnAF(dst1,dst2 *data.Slice) {

	// Set Effective field

	if (!isolatedlattices) {
	UpdateM()
	SetDemagField(dst1)
	data.Copy(dst2, dst1)
	} else {   // Just for code debug
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
	if !relaxing {
                if LLBeq!=true {
                  B_therm.AddToAF(dst1,dst2)
                 }
	}
	AddCustomField(dst1)
	AddCustomField(dst2)
	
	// Add to sublattice 1 and 2
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	if Precess {
		cuda.LLTorque(dst1, M1.Buffer(), dst1, alpha) // overwrite dst with torque
		cuda.LLTorque(dst2, M2.Buffer(), dst2, alpha) 
	} else {
		cuda.LLNoPrecess(dst1, M1.Buffer(), dst1)
		cuda.LLNoPrecess(dst2, M2.Buffer(), dst2)
	}
	
	// STT
	AddSTTorqueAF(dst1,dst2)

	FreezeSpins(dst1)
	FreezeSpins(dst2)

	NEvals++
}


func AddSTTorqueAF(dst1,dst2 *data.Slice) {

	if J.isZero() {
		return
	}
	util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
	jspin, rec := J.Slice()
	if rec {
		defer cuda.Recycle(jspin)
	}
	fl, rec := FixedLayer.Slice()
	if rec {
		defer cuda.Recycle(fl)
	}
	//Lattice 1 and 2
	if !DisableZhangLiTorque {
		msat1 := Msat1.MSlice()
		defer msat1.Recycle()
		msat2 := Msat2.MSlice()
		defer msat2.Recycle()
		j := J.MSlice()
		defer j.Recycle()
		alpha := Alpha.MSlice()
		defer alpha.Recycle()
		xi := Xi.MSlice()
		defer xi.Recycle()
		pol := Pol.MSlice()
		defer pol.Recycle()
		cuda.AddZhangLiTorque(dst1, M1.Buffer(), msat1, j, alpha, xi, pol, Mesh())
		cuda.AddZhangLiTorque(dst2, M2.Buffer(), msat2, j, alpha, xi, pol, Mesh())
	}
	if !DisableSlonczewskiTorque && !FixedLayer.isZero() {

			msat := Msat.MSlice()
			defer msat.Recycle()
			msat1 := Msat1.MSlice()
			defer msat1.Recycle()
			msat2 := Msat2.MSlice()
			defer msat2.Recycle()
			j := J.MSlice()
			defer j.Recycle()
			fixedP := FixedLayer.MSlice()
			defer fixedP.Recycle()
			alpha := Alpha.MSlice()
			defer alpha.Recycle()
			pol1 := Pol1.MSlice()
			defer pol1.Recycle()
			pol2 := Pol2.MSlice()
			defer pol2.Recycle()
			lambda := Lambda.MSlice()
			defer lambda.Recycle()
			epsPrime := EpsilonPrime.MSlice()
			defer epsPrime.Recycle()
			cuda.AddSlonczewskiTorque2(dst1, M1.Buffer(),
				msat1, j, fixedP, alpha, pol1, lambda, epsPrime, Mesh())
			cuda.AddSlonczewskiTorque2(dst2, M2.Buffer(),
				msat2, j, fixedP, alpha, pol2, lambda, epsPrime, Mesh())
	}
}



// Thermal field
func (b *thermField) AddToAF(dst1, dst2 *data.Slice) {
	if !Temp.isZero() {
		b.updateAF(1)
		cuda.Add(dst1, dst1, b.noise)
		b.updateAF(2)
		cuda.Add(dst2, dst2, b.noise)
	}
}

func (b *thermField) updateAF(i int) {
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
		return
	}

	// keep constant during time step
	if NSteps == b.step && Dt_si == b.dt && solvertype<6 {
		return
	}

	if FixDt == 0 {
		util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}

	N := Mesh().NCell()
	k2_VgammaDt := 2 * mag.Kb / (GammaLL * cellVolume() * Dt_si)
	noise := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(noise)

	const mean = 0
	const stddev = 1
	dst := b.noise
	
	ms:=Msat1.MSlice()
	defer ms.Recycle()
	if (i==1) {
		ms = Msat1.MSlice()
		defer ms.Recycle()
		}
	if (i==2) {
		ms = Msat2.MSlice()
		defer ms.Recycle()
		}
	
	temp := Temp.MSlice()
	defer temp.Recycle()
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	Noise_scale:=1.0
	if (JHThermalnoise==false){
	Noise_scale=0.0} else {
	Noise_scale=1.0
	}  // To cancel themal noise if needed
	for i := 0; i < 3; i++ {
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		cuda.SetTemperature(dst.Comp(i), noise, k2_VgammaDt*Noise_scale, ms, temp, alpha)
	}
	b.step = NSteps
	b.dt = Dt_si
}

func AddAnisotropyFieldAF(dst1,dst2 *data.Slice) {
	addUniaxialAnisotropyFrom(dst1, M1, Msat1, Ku11, Ku21, AnisU1)
	addUniaxialAnisotropyFrom(dst2, M2, Msat2, Ku12, Ku22, AnisU2)
	addCubicAnisotropyFrom(dst1, M1, Msat1, Kc11, Kc21, Kc31, AnisC11, AnisC21)
	addCubicAnisotropyFrom(dst2, M2, Msat2, Kc12, Kc22, Kc32, AnisC12, AnisC22)
}

func UpdateM() {
	ms0 := Msat.MSlice()
	defer ms0.Recycle()
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	cuda.NormalizeAF(M.Buffer(),M1.Buffer(),M2.Buffer(), ms0,ms1,ms2)
}
