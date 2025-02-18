package engine

// Antiferromagnetic codes. Includes definitios adn changes to AF torques and thermal field

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	"github.com/mumax/3/util"
)

// Anisotropy, magnetization, exchange... variables and magenetization for sublattices
var (
	Ku11 = NewScalarParam("Ku11", "J/m3", "1st order uniaxial anisotropy constant")
	Ku21 = NewScalarParam("Ku21", "J/m3", "1st order uniaxial anisotropy constant")
	Ku12 = NewScalarParam("Ku12", "J/m3", "1st order uniaxial anisotropy constant")
	Ku22 = NewScalarParam("Ku22", "J/m3", "1st order uniaxial anisotropy constant")

	// Second set for another uniaxial Anisotropy
	Ku11b   = NewScalarParam("Ku11b", "J/m3", "1st order uniaxial anisotropy constant")
	Ku21b   = NewScalarParam("Ku21b", "J/m3", "1st order uniaxial anisotropy constant")
	Ku12b   = NewScalarParam("Ku12b", "J/m3", "1st order uniaxial anisotropy constant")
	Ku22b   = NewScalarParam("Ku22b", "J/m3", "1st order uniaxial anisotropy constant")
	AnisU1b = NewVectorParam("anisU1b", "", "Uniaxial anisotropy direction")

	Kc11             = NewScalarParam("Kc11", "J/m3", "1st order cubic anisotropy constant")
	Kc21             = NewScalarParam("Kc21", "J/m3", "2nd order cubic anisotropy constant")
	Kc31             = NewScalarParam("Kc31", "J/m3", "3rd order cubic anisotropy constant")
	Kc12             = NewScalarParam("Kc12", "J/m3", "1st order cubic anisotropy constant")
	Kc22             = NewScalarParam("Kc22", "J/m3", "2nd order cubic anisotropy constant")
	Kc32             = NewScalarParam("Kc32", "J/m3", "3rd order cubic anisotropy constant")
	AnisU1           = NewVectorParam("anisU1", "", "Uniaxial anisotropy direction")
	AnisC11          = NewVectorParam("anisC11", "", "Cubic anisotropy direction #1")
	AnisC21          = NewVectorParam("anisC21", "", "Cubic anisotorpy directon #2")
	AnisU2           = NewVectorParam("anisU2", "", "Uniaxial anisotropy direction")
	AnisC12          = NewVectorParam("anisC12", "", "Cubic anisotropy direction #1")
	AnisC22          = NewVectorParam("anisC22", "", "Cubic anisotorpy directon #2")
	M1               magnetization // reduced magnetization (unit length)
	M2               magnetization // reduced magnetization (unit length)
	Msat1                          = NewScalarParam("Msat1", "A/m", "Saturation magnetization")
	M_full1                        = NewVectorField("m_full1", "A/m", "Unnormalized magnetization", SetMFull1)
	Msat2                          = NewScalarParam("Msat2", "A/m", "Saturation magnetization")
	M_full2                        = NewVectorField("m_full2", "A/m", "Unnormalized magnetization", SetMFull2)
	GammaLL1         float64       = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	GammaLL2         float64       = 1.7595e11 // Gyromagnetic ratio of spins, in rad/Ts
	Pol1                           = NewScalarParam("Pol1", "", "Electrical current polarization Zhang-Li Lattice 1")
	Pol2                           = NewScalarParam("Pol2", "", "Electrical current polarization Zhang-Li Lattice 2")
	isolatedlattices bool          = false // Debug only
	Alpha1                         = NewScalarParam("alpha1", "", "Landau-Lifshitz damping constant")
	Alpha2                         = NewScalarParam("alpha2", "", "Landau-Lifshitz damping constant")
	Xi1                            = NewScalarParam("xi1", "", "Non-adiabaticity of spin-transfer-torque STT lattice 1")
	Xi2                            = NewScalarParam("xi2", "", "Non-adiabaticity of spin-transfer-torque STT lattice 2")
	PolSTT1                        = NewScalarParam("PolSTT1", "", "Electrical current polarization STT lattice 1")
	PolSTT2                        = NewScalarParam("PolSTT2", "", "Electrical current polarization STT lattice 2")
	// For AF PRB 054401
	x_TM = NewScalarParam("x_TM", "a.u.", "TM ratio")
	nv   = NewScalarParam("nv", "a.u.", "Number of neighbours")
	mu1  = NewScalarParam("mu1", "J/T.", "Bohr magnetons lattice 1")
	mu2  = NewScalarParam("mu2", "J/T.", "Bohr magnetons lattice 2")
	J0aa = NewScalarParam("J0aa", "T.", "Exchange lattice 1")
	J0bb = NewScalarParam("J0bb", "T.", "Exchange lattice 2")
	J0ab = NewScalarParam("J0ab", "T.", "Exchange lattice 1-2")

	EpsilonPrime1 = NewScalarParam("EpsilonPrime1", "", "Slonczewski secondairy STT term ε' Lattice 1")
	EpsilonPrime2 = NewScalarParam("EpsilonPrime2", "", "Slonczewski secondairy STT term ε' Lattice 2")

	// For LLB AF Angular momentum exchange (Unai)
	lambda0 = NewScalarParam("lambda0", "", "Moment exchange between sublattices")
	MFA     = false
	// For Brillouin
	Brillouin = false
	JA        = NewScalarParam("JA", "a.u.", "Billouin J lattice A")
	JB        = NewScalarParam("JB", "a.u.", "Billouin J lattice B")
	// Direct moment induction
	deltaM = NewScalarParam("deltaM", "a.u.", "Moment indiced by laser")

	// For magnetoelastic
	B11        = NewScalarParam("B11", "J/m3", "First magneto-elastic coupling constant subnet 1")
	B21        = NewScalarParam("B21", "J/m3", "Second magneto-elastic coupling constantsubnet 1")
	B_mel1     = NewVectorField("B_mel1", "T", "Magneto-elastic filed subnet 1", AddMagnetoelasticField1)
	F_mel1     = NewVectorField("F_mel1", "N/m3", "Magneto-elastic force density subnet 1", GetMagnetoelasticForceDensity1)
	Edens_mel1 = NewScalarField("Edens_mel1", "J/m3", "Magneto-elastic energy density subnet 1", AddMagnetoelasticEnergyDensity1)
	E_mel1     = NewScalarValue("E_mel1", "J", "Magneto-elastic energy", GetMagnetoelasticEnergy1)

	B12        = NewScalarParam("B12", "J/m3", "First magneto-elastic coupling constant subnet 2")
	B22        = NewScalarParam("B22", "J/m3", "Second magneto-elastic coupling constant subnet 2")
	B_mel2     = NewVectorField("B_mel2", "T", "Magneto-elastic filed subnet 2", AddMagnetoelasticField2)
	F_mel2     = NewVectorField("F_mel2", "N/m3", "Magneto-elastic force density subnet 2", GetMagnetoelasticForceDensity2)
	Edens_mel2 = NewScalarField("Edens_mel2", "J/m3", "Magneto-elastic energy density subnet 2", AddMagnetoelasticEnergyDensity2)
	E_mel2     = NewScalarValue("E_mel2", "J", "Magneto-elastic energy subnet 2", GetMagnetoelasticEnergy2)

	// M and neel vectors
	MAFg  = NewVectorField("MAFg", "A/m", "Moment AF", GmagnetizationAF)
	Neelg = NewVectorField("Neelg", "A/m", "Neel moment AF", GneelAF)
	MAF   = NewVectorField("MAF", "A/m", "Magnetization AF", MagnetizationAF)
	Neel  = NewVectorField("Neel", "A/m", "Magnetization AF", NeelAF)
)

func init() {
	DeclFunc("UpdateM", UpdateM, "UpdateM")
	DeclFunc("InitAntiferro", InitAntiferro, "InitAntiferro")
	DeclLValue("m1", &M1, `Reduced magnetization sublattice 1 (unit length)`)
	DeclLValue("m2", &M2, `Reduced magnetization sublattice 2 (unit length)`)
	M1.name = "m1_"
	M2.name = "m2_"
	DeclVar("GammaLL1", &GammaLL1, "Gyromagnetic ratio in rad/Ts Lattice 1")
	DeclVar("GammaLL2", &GammaLL2, "Gyromagnetic ratio in rad/Ts Lattice 2")
	DeclVar("isolatedlattices", &isolatedlattices, "Isolate AF lattices")
	DeclVar("MFA", &MFA, "MFA model for AF LLB")
	DeclVar("Brillouin", &Brillouin, "Brillouin model for AF LLB")
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
func torqueFnAF(dst1, dst2 *data.Slice) {

	// Set Effective field

	if !isolatedlattices {
		UpdateM()
		SetDemagField(dst1)
		data.Copy(dst2, dst1)
	} else { // Just for code debug
		data.Copy(M.Buffer(), M1.Buffer())
		*Msat = *Msat1
		SetDemagField(dst1)
		data.Copy(M.Buffer(), M2.Buffer())
		*Msat = *Msat2
		SetDemagField(dst2)
	}
	AddExchangeFieldAF(dst1, dst2)
	AddAnisotropyFieldAF(dst1, dst2)
	AddMagnetoelasticField1(dst1)
	AddMagnetoelasticField2(dst2)
	//AddAFMExchangeField(dst)  // AFM Exchange non adjacent layers
	B_ext.AddTo(dst1)
	B_ext.AddTo(dst2)
	if !relaxing {
		if LLBeq != true {
			B_therm.AddToAF(dst1, dst2)
		}
	}
	AddCustomField(dst1)
	AddCustomField(dst2)

	// Add to sublattice 1 and 2
	alpha1 := Alpha1.MSlice()
	defer alpha1.Recycle()
	alpha2 := Alpha2.MSlice()
	defer alpha2.Recycle()
	if Precess {
		cuda.LLTorque(dst1, M1.Buffer(), dst1, alpha1) // overwrite dst with torque
		cuda.LLTorque(dst2, M2.Buffer(), dst2, alpha2)
	} else {
		cuda.LLNoPrecess(dst1, M1.Buffer(), dst1)
		cuda.LLNoPrecess(dst2, M2.Buffer(), dst2)
	}

	// STT
	AddSTTorqueAF(dst1, dst2)

	FreezeSpins(dst1)
	FreezeSpins(dst2)

	NEvals++
}

func AddSTTorqueAF(dst1, dst2 *data.Slice) {

	if J.isZero() {
		return
	}
	//util.AssertMsg(!Pol.isZero(), "spin polarization should not be 0")
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
		alpha1 := Alpha1.MSlice()
		defer alpha1.Recycle()
		alpha2 := Alpha2.MSlice()
		defer alpha2.Recycle()
		xi1 := Xi1.MSlice()
		defer xi1.Recycle()
		xi2 := Xi2.MSlice()
		defer xi2.Recycle()
		polstt1 := PolSTT1.MSlice()
		defer polstt1.Recycle()
		polstt2 := PolSTT2.MSlice()
		defer polstt2.Recycle()
		cuda.AddZhangLiTorque(dst1, M1.Buffer(), msat1, j, alpha1, xi1, polstt1, Mesh())
		cuda.AddZhangLiTorque(dst2, M2.Buffer(), msat2, j, alpha2, xi2, polstt2, Mesh())
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
		alpha1 := Alpha1.MSlice()
		defer alpha1.Recycle()
		alpha2 := Alpha2.MSlice()
		defer alpha2.Recycle()
		pol1 := Pol1.MSlice()
		defer pol1.Recycle()
		pol2 := Pol2.MSlice()
		defer pol2.Recycle()
		lambda := Lambda.MSlice()
		defer lambda.Recycle()
		//epsPrime := EpsilonPrime.MSlice()
		//defer epsPrime.Recycle()
		epsPrime1 := EpsilonPrime1.MSlice()
		defer epsPrime1.Recycle()
		epsPrime2 := EpsilonPrime2.MSlice()
		defer epsPrime2.Recycle()
		thickness := FreeLayerThickness.MSlice()
		defer thickness.Recycle()
		if LLBeq == false {
			cuda.AddSlonczewskiTorque2(dst1, M1.Buffer(),
				msat1, j, fixedP, alpha1, pol1, lambda, epsPrime1,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
			cuda.AddSlonczewskiTorque2(dst2, M2.Buffer(),
				msat2, j, fixedP, alpha2, pol2, lambda, epsPrime2,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
		} else {
			cuda.AddSlonczewskiTorque2LLB(dst1, M1.Buffer(),
				msat1, j, fixedP, alpha1, pol1, lambda, epsPrime1,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
			cuda.AddSlonczewskiTorque2LLB(dst2, M2.Buffer(),
				msat2, j, fixedP, alpha2, pol2, lambda, epsPrime2,
				thickness,
				CurrentSignFromFixedLayerPosition[fixedLayerPosition],
				Mesh())
		}

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
	if NSteps == b.step && Dt_si == b.dt && solvertype < 6 {
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

	ms := Msat1.MSlice()
	defer ms.Recycle()
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	if i == 1 {
		ms = Msat1.MSlice()
		defer ms.Recycle()
		alpha = Alpha1.MSlice()
	}
	if i == 2 {
		ms = Msat2.MSlice()
		defer ms.Recycle()
		alpha = Alpha2.MSlice()
	}

	temp := Temp.MSlice()
	defer temp.Recycle()
	//alpha0 := Alpha.MSlice()
	//defer alpha0.Recycle()
	Noise_scale := 1.0
	if JHThermalnoise == false {
		Noise_scale = 0.0
	} else {
		Noise_scale = 1.0
	} // To cancel themal noise if needed
	for i := 0; i < 3; i++ {
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		cuda.SetTemperature(dst.Comp(i), noise, k2_VgammaDt*Noise_scale, ms, temp, alpha, ScaleNoiseLLB)
	}
	b.step = NSteps
	b.dt = Dt_si
}

func AddAnisotropyFieldAF(dst1, dst2 *data.Slice) {
	addUniaxialAnisotropyFrom(dst1, M1, Msat1, Ku11, Ku21, AnisU1)
	addUniaxialAnisotropyFrom(dst2, M2, Msat2, Ku12, Ku22, AnisU2)
	// second axis
	addUniaxialAnisotropyFrom(dst1, M1, Msat1, Ku11b, Ku21b, AnisU1b)
	addUniaxialAnisotropyFrom(dst2, M2, Msat2, Ku12b, Ku22b, AnisU1b)

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
	cuda.NormalizeAF(M.Buffer(), M1.Buffer(), M2.Buffer(), ms0, ms1, ms2)
}

func MagnetizationAF(dst *data.Slice) {
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	cuda.MagnetizationAF(dst, M1.Buffer(), M2.Buffer(), ms1, ms2)
}

func NeelAF(dst *data.Slice) {
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	cuda.NeelAF(dst, M1.Buffer(), M2.Buffer(), ms1, ms2)
}

func GmagnetizationAF(dst *data.Slice) {
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	cuda.GMagnetizationAF(dst, M1.Buffer(), M2.Buffer(), ms1, ms2, float32(GammaLL1), float32(GammaLL2))
}

func GneelAF(dst *data.Slice) {
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	cuda.GNeelAF(dst, M1.Buffer(), M2.Buffer(), ms1, ms2, float32(GammaLL1), float32(GammaLL2))
}

func RenormAF(y01, y02 *data.Slice, dt, GammaLL1, GammaLL2 float32) {

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

	cuda.LLBRenormAF(y01, y02, M1.Buffer(), M2.Buffer(), temp, alpha, alpha1, alpha2, Tcurie, Msat, Msat1, Msat2, X_TM, NV, MU1, MU2, J0AA, J0BB, J0AB, dt, GammaLL1, GammaLL2)

}

func RenormAFBri(y01, y02 *data.Slice, dt, GammaLL1, GammaLL2 float32) {

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

	// for Brillouin
	Ja := JA.MSlice()
	defer Ja.Recycle()
	Jb := JB.MSlice()
	defer Jb.Recycle()

	cuda.LLBRenormAFBri(y01, y02, M1.Buffer(), M2.Buffer(), temp, alpha, alpha1, alpha2, Tcurie, Msat, Msat1, Msat2, X_TM, NV, MU1, MU2, J0AA, J0BB, J0AB, dt, GammaLL1, GammaLL2, Ja, Jb)

}

// Magnetoelastic functions for subnet 1 and 2

//Subnet 1
func AddMagnetoelasticField1(dst *data.Slice) {
	haveMel := B11.nonZero() || B21.nonZero()
	if !haveMel {
		return
	}

	Exx := exx.MSlice()
	defer Exx.Recycle()

	Eyy := eyy.MSlice()
	defer Eyy.Recycle()

	Ezz := ezz.MSlice()
	defer Ezz.Recycle()

	Exy := exy.MSlice()
	defer Exy.Recycle()

	Exz := exz.MSlice()
	defer Exz.Recycle()

	Eyz := eyz.MSlice()
	defer Eyz.Recycle()

	b11 := B11.MSlice()
	defer b11.Recycle()

	b21 := B21.MSlice()
	defer b21.Recycle()

	ms1 := Msat1.MSlice()
	defer ms1.Recycle()

	cuda.AddMagnetoelasticField(dst, M1.Buffer(),
		Exx, Eyy, Ezz,
		Exy, Exz, Eyz,
		b11, b21, ms1)
}

func GetMagnetoelasticForceDensity1(dst *data.Slice) {
	haveMel := B11.nonZero() || B21.nonZero()
	if !haveMel {
		return
	}

	util.AssertMsg(B11.IsUniform() && B21.IsUniform(), "Magnetoelastic: B11, B21 must be uniform")

	b11 := B11.MSlice()
	defer b11.Recycle()

	b21 := B21.MSlice()
	defer b21.Recycle()

	cuda.GetMagnetoelasticForceDensity(dst, M1.Buffer(),
		b11, b21, M1.Mesh())
}

func AddMagnetoelasticEnergyDensity1(dst *data.Slice) {
	haveMel := B11.nonZero() || B21.nonZero()
	if !haveMel {
		return
	}

	buf := cuda.Buffer(B_mel1.NComp(), B_mel1.Mesh().Size())
	defer cuda.Recycle(buf)

	// unnormalized magnetization:
	Mf1 := ValueOf(M_full1)
	defer cuda.Recycle(Mf1)

	Exx := exx.MSlice()
	defer Exx.Recycle()

	Eyy := eyy.MSlice()
	defer Eyy.Recycle()

	Ezz := ezz.MSlice()
	defer Ezz.Recycle()

	Exy := exy.MSlice()
	defer Exy.Recycle()

	Exz := exz.MSlice()
	defer Exz.Recycle()

	Eyz := eyz.MSlice()
	defer Eyz.Recycle()

	b11 := B11.MSlice()
	defer b11.Recycle()

	b21 := B21.MSlice()
	defer b21.Recycle()

	ms1 := Msat1.MSlice()
	defer ms1.Recycle()

	zeromel := zeroMel.MSlice()
	defer zeromel.Recycle()

	// 1st
	cuda.Zero(buf)
	cuda.AddMagnetoelasticField(buf, M1.Buffer(),
		Exx, Eyy, Ezz,
		Exy, Exz, Eyz,
		b11, zeromel, ms1)
	cuda.AddDotProduct(dst, -1./2., buf, Mf1)

	// 1nd
	cuda.Zero(buf)
	cuda.AddMagnetoelasticField(buf, M1.Buffer(),
		Exx, Eyy, Ezz,
		Exy, Exz, Eyz,
		zeromel, b21, ms1)
	cuda.AddDotProduct(dst, -1./1., buf, Mf1)
}

// Returns magneto-ell energy in joules.
func GetMagnetoelasticEnergy1() float64 {
	haveMel := B11.nonZero() || B21.nonZero()
	if !haveMel {
		return float64(0.0)
	}

	buf := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(buf)

	cuda.Zero(buf)
	AddMagnetoelasticEnergyDensity1(buf)
	return cellVolume() * float64(cuda.Sum(buf))
}

// Subnet 2
func AddMagnetoelasticField2(dst *data.Slice) {
	haveMel := B12.nonZero() || B22.nonZero()
	if !haveMel {
		return
	}

	Exx := exx.MSlice()
	defer Exx.Recycle()

	Eyy := eyy.MSlice()
	defer Eyy.Recycle()

	Ezz := ezz.MSlice()
	defer Ezz.Recycle()

	Exy := exy.MSlice()
	defer Exy.Recycle()

	Exz := exz.MSlice()
	defer Exz.Recycle()

	Eyz := eyz.MSlice()
	defer Eyz.Recycle()

	b12 := B12.MSlice()
	defer b12.Recycle()

	b22 := B22.MSlice()
	defer b22.Recycle()

	ms2 := Msat2.MSlice()
	defer ms2.Recycle()

	cuda.AddMagnetoelasticField(dst, M2.Buffer(),
		Exx, Eyy, Ezz,
		Exy, Exz, Eyz,
		b12, b22, ms2)
}

func GetMagnetoelasticForceDensity2(dst *data.Slice) {
	haveMel := B12.nonZero() || B22.nonZero()
	if !haveMel {
		return
	}

	util.AssertMsg(B12.IsUniform() && B22.IsUniform(), "Magnetoelastic: B12, B22 must be uniform")

	b12 := B2.MSlice()
	defer b12.Recycle()

	b22 := B22.MSlice()
	defer b22.Recycle()

	cuda.GetMagnetoelasticForceDensity(dst, M2.Buffer(),
		b12, b22, M2.Mesh())
}

func AddMagnetoelasticEnergyDensity2(dst *data.Slice) {
	haveMel := B12.nonZero() || B22.nonZero()
	if !haveMel {
		return
	}

	buf := cuda.Buffer(B_mel2.NComp(), B_mel2.Mesh().Size())
	defer cuda.Recycle(buf)

	// unnormalized magnetization:
	Mf2 := ValueOf(M_full2)
	defer cuda.Recycle(Mf2)

	Exx := exx.MSlice()
	defer Exx.Recycle()

	Eyy := eyy.MSlice()
	defer Eyy.Recycle()

	Ezz := ezz.MSlice()
	defer Ezz.Recycle()

	Exy := exy.MSlice()
	defer Exy.Recycle()

	Exz := exz.MSlice()
	defer Exz.Recycle()

	Eyz := eyz.MSlice()
	defer Eyz.Recycle()

	b12 := B12.MSlice()
	defer b12.Recycle()

	b22 := B22.MSlice()
	defer b22.Recycle()

	ms2 := Msat2.MSlice()
	defer ms2.Recycle()

	zeromel := zeroMel.MSlice()
	defer zeromel.Recycle()

	// 1st
	cuda.Zero(buf)
	cuda.AddMagnetoelasticField(buf, M2.Buffer(),
		Exx, Eyy, Ezz,
		Exy, Exz, Eyz,
		b12, zeromel, ms2)
	cuda.AddDotProduct(dst, -1./2., buf, Mf2)

	// 1nd
	cuda.Zero(buf)
	cuda.AddMagnetoelasticField(buf, M2.Buffer(),
		Exx, Eyy, Ezz,
		Exy, Exz, Eyz,
		zeromel, b22, ms2)
	cuda.AddDotProduct(dst, -1./1., buf, Mf2)
}

// Returns magneto-ell energy in joules.
func GetMagnetoelasticEnergy2() float64 {
	haveMel := B12.nonZero() || B22.nonZero()
	if !haveMel {
		return float64(0.0)
	}

	buf := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(buf)

	cuda.Zero(buf)
	AddMagnetoelasticEnergyDensity2(buf)
	return cellVolume() * float64(cuda.Sum(buf))
}
