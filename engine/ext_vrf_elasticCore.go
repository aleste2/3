package engine

// Elastic solver Core

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Epecemos usando tipos definidos y ya veremos luego. Inicialmente 2D
var (
	C11       = NewScalarExcitation("C11", "", "Elastic parameter C11")
	C12       = NewScalarExcitation("C12", "", "Elastic parameter C12")
	C44       = NewScalarExcitation("C44", "", "Elastic parameter C44")
	Rho       = NewScalarExcitation("rho", "Kg/m3", "Mass density")
	Force     = NewScalarExcitation("force", "N/m3", "Force density")
	Eta       = NewScalarExcitation("eta", "", "Damping elastic")
	R         magnetization // displacement (m)
	U         magnetization // speed (m/s)
	Sigma     magnetization // sigma (xx,yy xy) // 2D by now
	B_ME      = NewVectorField("B_ME", "T", "Dynamic Magneto-elastic field", AddMEField)
	Strain    = NewVectorField("Strain", "", "Dynamic Strain", AddStrainField)
	Edens_mME = NewScalarField("Edens_mME", "J/m3", "mME energy density", AddmMEEnergyDensity)
	E_mME     = NewScalarValue("E_mME", "J", "mME energy", GetmMEEnergy)
)

func init() {
	registerEnergy(GetmMEEnergy, AddmMEEnergyDensity)
	DeclLValue("r", &R, `Displacement (m)`)
	DeclLValue("u", &U, `Speed (m/s)`)
	DeclLValue("sigma", &Sigma, `Stress`)
	R.name = "r"
	U.name = "u"
	Sigma.name = "sigma"
	DeclFunc("InitME", InitME, "InitME")
}

func InitME() {
	R.alloc()
	cuda.Zero(R.Buffer())
	U.alloc()
	cuda.Zero(U.Buffer())
	Sigma.alloc()
	cuda.Zero(Sigma.Buffer())
}

type ElasticEuler struct{}

// Euler method, can be used as solver.Step.
func (_ *ElasticEuler) Step() {
	r0 := R.Buffer()
	u0 := U.Buffer()
	sigma0 := Sigma.Buffer()

	Dt_si = FixDt

	du := cuda.Buffer(VECTOR, u0.Size())
	defer cuda.Recycle(du)
	dsigma := cuda.Buffer(VECTOR, sigma0.Size())
	defer cuda.Recycle(dsigma)

	dt := float32(Dt_si)

	Calc_du(du)
	cuda.Madd2(u0, u0, du, 1, dt) // v = v + dt * dv
	cuda.Madd2(r0, r0, u0, 1, dt) // x = x + dt * v
	Calc_dsigma(dsigma)
	cuda.Madd2(sigma0, sigma0, dsigma, 1, dt) // s = s + dt * ds

	Time += Dt_si
	NSteps++
}

func (_ *ElasticEuler) Free() {}

type ElasticRK4 struct{}

// Euler method, can be used as solver.Step.
func (_ *ElasticRK4) Step() {
	// RK4

	//m := M.Buffer()
	//size := m.Size()
	r := R.Buffer()
	u := U.Buffer()
	sigma := Sigma.Buffer()
	size := r.Size()

	Dt_si = FixDt

	t0 := Time
	// backup values
	r0 := cuda.Buffer(3, size)
	defer cuda.Recycle(r0)
	data.Copy(r0, r)
	u0 := cuda.Buffer(3, size)
	defer cuda.Recycle(u0)
	data.Copy(u0, u)
	sigma0 := cuda.Buffer(3, size)
	defer cuda.Recycle(sigma0)
	data.Copy(sigma0, sigma)

	kr1, kr2, kr3, kr4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	ku1, ku2, ku3, ku4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	ksigma1, ksigma2, ksigma3, ksigma4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(kr1)
	defer cuda.Recycle(kr2)
	defer cuda.Recycle(kr3)
	defer cuda.Recycle(kr4)
	defer cuda.Recycle(ku1)
	defer cuda.Recycle(ku2)
	defer cuda.Recycle(ku3)
	defer cuda.Recycle(ku4)
	defer cuda.Recycle(ksigma1)
	defer cuda.Recycle(ksigma2)
	defer cuda.Recycle(ksigma3)
	defer cuda.Recycle(ksigma4)

	h := float32(Dt_si) // time step = Dt_Si

	// stage 1
	//torqueFn(k1)
	Calc_du(ku1)
	Calc_dsigma(ksigma1)
	//cuda.Madd2(kr1, kr1, u, 0, dt) // dx = dt * v0?

	// stage 2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(u, u, ku1, 1, (1./2.)*h)
	//cuda.Madd2(r, r, kr1, 1, (1./2.)*h)
	cuda.Madd2(sigma, sigma, ksigma1, 1, (1./2.)*h)
	//torqueFn(k2)
	Calc_du(ku2)
	Calc_dsigma(ksigma2)
	//cuda.Madd2(kr2, kr2, u, 0, dt) // dx = dt * v0?

	// stage 3
	//cuda.Madd2(m, m0, k2, 1, (1./2.)*h) // m = m0*1 + k2*1/2
	cuda.Madd2(u, u0, ku2, 1, (1./2.)*h)
	cuda.Madd2(sigma, sigma0, ksigma2, 1, (1./2.)*h)
	//M.normalize()
	//torqueFn(k3)
	Calc_du(ku3)
	Calc_dsigma(ksigma3)

	// stage 4
	Time = t0 + Dt_si
	//cuda.Madd2(m, m0, k3, 1, 1.*h) // m = m0*1 + k3*1
	cuda.Madd2(u, u0, ku3, 1, 1.*h)             // m = m0*1 + k3*1
	cuda.Madd2(sigma, sigma0, ksigma3, 1, 1.*h) // m = m0*1 + k3*1
	//M.normalize()
	//torqueFn(k4)
	Calc_du(ku4)
	Calc_dsigma(ksigma4)

	//cuda.Madd5(m, m0, k1, k2, k3, k4, 1, (1./6.)*h, (1./3.)*h, (1./3.)*h, (1./6.)*h)
	cuda.Madd5(u, u0, ku1, ku2, ku3, ku4, 1, (1./6.)*h, (1./3.)*h, (1./3.)*h, (1./6.)*h)
	cuda.Madd5(sigma, sigma0, ksigma1, ksigma2, ksigma3, ksigma4, 1, (1./6.)*h, (1./3.)*h, (1./3.)*h, (1./6.)*h)

	cuda.Madd2(r, r, u, 1, h) // x = x + dt * v

}

func (_ *ElasticRK4) Free() {}

// Magnetoelastic solver
type MagElasticEuler struct {
	mold   *data.Slice // m at previous step
	deltat float32
}

// Euler method, can be used as solver.Step.
func (EulerME *MagElasticEuler) Step() {
	m := M.Buffer()
	size := m.Size()
	m_0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m_0)
	data.Copy(m_0, m)

	// upon resize: remove wrongly sized k1
	//if EulerME.mold.Size() != size {
	//	EulerME.Free()
	//	EulerME.mold = nil
	//}
	// first step ever: one-time k1 init and eval
	if EulerME.mold == nil {
		EulerME.mold = cuda.NewSlice(3, size)
		data.Copy(EulerME.mold, m)
		EulerME.deltat = float32(FixDt)
	}

	m0 := M.Buffer()
	r0 := R.Buffer()
	u0 := U.Buffer()
	sigma0 := Sigma.Buffer()

	Dt_si = FixDt

	dm := cuda.Buffer(VECTOR, m0.Size())
	defer cuda.Recycle(dm)
	du := cuda.Buffer(VECTOR, u0.Size())
	defer cuda.Recycle(du)
	dsigma := cuda.Buffer(VECTOR, sigma0.Size())
	defer cuda.Recycle(dsigma)

	dt := float32(Dt_si)
	dtm := float32(Dt_si * GammaLL)
	/*
		Calc_du(du)
		cuda.Madd2(u0, u0, du, 1, dt) // v = v + dt * dv
		cuda.Madd2(r0, r0, u0, 1, dt) // x = x + dt * v
		Calc_dsigmam(dsigma, EulerME.mold, float32(EulerME.deltat))
		cuda.Madd2(sigma0, sigma0, dsigma, 1, dt) // s = s + dt * ds
		torqueFn(dm)
		setMaxTorque(dm)
		cuda.Madd2(m0, m0, dm, 1, dtm) // y = y + dt * dy
		M.normalize()
	*/

	torqueFn(dm)
	setMaxTorque(dm)
	cuda.Madd2(m0, m0, dm, 1, dtm) // y = y + dt * dy
	M.normalize()
	Calc_du(du)
	cuda.Madd2(u0, u0, du, 1, dt) // v = v + dt * dv
	cuda.Madd2(r0, r0, u0, 1, dt) // x = x + dt * v
	Calc_dsigmam(dsigma, dm, float32(1.0/GammaLL))
	cuda.Madd2(sigma0, sigma0, dsigma, 1, dt) // s = s + dt * ds

	Time += Dt_si
	data.Copy(EulerME.mold, m_0)
	EulerME.deltat = float32(FixDt)
	NSteps++

}

func (EulerMe *MagElasticEuler) Free() {
	EulerMe.mold.Free()
	EulerMe.mold = nil
}

// Etoelastic calculators
func Calc_du(dst *data.Slice) {
	eta := Eta.MSlice()
	defer eta.Recycle()
	rho := Rho.MSlice()
	defer rho.Recycle()
	force := Force.MSlice()
	defer force.Recycle()

	u := U.Buffer()
	sigma := Sigma.Buffer()

	cuda.CalcDU(dst, sigma, u, eta, rho, force, M.Mesh())
	NEvals++
}

func Calc_dsigma(dst *data.Slice) {
	c11 := C11.MSlice()
	defer c11.Recycle()
	c12 := C12.MSlice()
	defer c12.Recycle()
	c44 := C44.MSlice()
	defer c44.Recycle()

	u := U.Buffer()

	cuda.CalcDSigma(dst, u, c11, c12, c44, M.Mesh())
	NEvals++
}

// Magnetoelastic calculators
func Calc_dsigmam(dst, mold *data.Slice, deltat float32) {
	c11 := C11.MSlice()
	defer c11.Recycle()
	c12 := C12.MSlice()
	defer c12.Recycle()
	c44 := C44.MSlice()
	defer c44.Recycle()
	b1 := B1.MSlice()
	defer b1.Recycle()
	b2 := B2.MSlice()
	defer b2.Recycle()

	u := U.Buffer()
	m := M.Buffer()
	//mold := EulerMe.mold
	//deltat := EulerMe.deltat

	cuda.CalcDSigmam(dst, u, c11, c12, c44, b1, b2, M.Mesh(), m, mold, deltat)
	NEvals++
}

// ME Field

func AddMEField(dst *data.Slice) {
	haveMel := B1.nonZero() || B2.nonZero()
	if !haveMel {
		return
	}

	c11 := C11.MSlice()
	defer c11.Recycle()
	c12 := C12.MSlice()
	defer c12.Recycle()
	c44 := C44.MSlice()
	defer c44.Recycle()
	b1 := B1.MSlice()
	defer b1.Recycle()
	b2 := B2.MSlice()
	defer b2.Recycle()
	ms := Msat.MSlice()
	defer ms.Recycle()

	//	cuda.AddMEField2(dst, M.Buffer(), R.Buffer(),
	cuda.AddMEField2(dst, M.Buffer(), Sigma.Buffer(),
		c11, c12, c44,
		b1, b2, ms, M.Mesh())
}

func AddStrainField(dst *data.Slice) {
	cuda.AddStrain(dst, R.Buffer(), M.Mesh())
}

func AddmMEEnergyDensity(dst *data.Slice) {
	buf := cuda.Buffer(B_ME.NComp(), Mesh().Size())
	defer cuda.Recycle(buf)

	// unnormalized magnetization:
	Mf := ValueOf(M_full)
	defer cuda.Recycle(Mf)

	cuda.Zero(buf)
	AddMEField(buf)
	cuda.AddDotProduct(dst, -1./2., buf, Mf)

}

func GetmMEEnergy() float64 {
	buf := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(buf)

	cuda.Zero(buf)
	AddmMEEnergyDensity(buf)
	return cellVolume() * float64(cuda.Sum(buf))
}
