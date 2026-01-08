package engine

// Elastic solver Core

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
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
	Edens_kin = NewScalarField("Edens_kin", "J/m3", "Kinetic energy density", GetKineticEnergy)
	E_kin     = NewScalarValue("E_kin", "J", "Kinetic energy", GetTotKineticEnergy)
	Edens_el  = NewScalarField("Edens_el", "J/m3", "Elastic energy density", GetElasticEnergy)
	E_el      = NewScalarValue("E_el", "J", "Elastic energy", GetTotElasticEnergy)

	// To test
	MaxTorqueSigma = NewScalarValue("maxTorqueSigma", "au", "Maximum torque Sigma, over all cells", GetMaxTorqueSigma)
	MaxTorqueSpped = NewScalarValue("maxTorqueSpeed", "au", "Maximum torque Speed, over all cells", GetMaxTorqueSpeed)
)

func init() {
	//registerEnergy(GetmMEEnergy, AddmMEEnergyDensity)
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

	NSteps++

}

func (_ *ElasticRK4) Free() {}

// RK4 solver
type MERK4s struct {
}

func (rk *MERK4s) Step() {
	m := M.Buffer()
	size := m.Size()
	r := R.Buffer()
	u := U.Buffer()
	sigma := Sigma.Buffer()

	if FixDt != 0 {
		Dt_si = FixDt
	}

	t0 := Time
	// backup magnetization
	m0 := cuda.Buffer(3, size)
	defer cuda.Recycle(m0)
	data.Copy(m0, m)
	r0 := cuda.Buffer(3, size)
	defer cuda.Recycle(r0)
	data.Copy(r0, r)
	u0 := cuda.Buffer(3, size)
	defer cuda.Recycle(u0)
	data.Copy(u0, u)
	sigma0 := cuda.Buffer(3, size)
	defer cuda.Recycle(sigma0)
	data.Copy(sigma0, sigma)

	k1, k2, k3, k4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	ku1, ku2, ku3, ku4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)
	ksigma1, ksigma2, ksigma3, ksigma4 := cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size), cuda.Buffer(3, size)

	defer cuda.Recycle(k1)
	defer cuda.Recycle(k2)
	defer cuda.Recycle(k3)
	defer cuda.Recycle(k4)

	defer cuda.Recycle(ku1)
	defer cuda.Recycle(ku2)
	defer cuda.Recycle(ku3)
	defer cuda.Recycle(ku4)

	defer cuda.Recycle(ksigma1)
	defer cuda.Recycle(ksigma2)
	defer cuda.Recycle(ksigma3)
	defer cuda.Recycle(ksigma4)

	h := float32(Dt_si * GammaLL) // internal time step = Dt dfor sigma,v
	h0 := float32(Dt_si)          // internal time step = Dt * gammaLL for m

	escala := float32(GammaLL)
	// stage 1
	torqueFn(k1)
	Calc_du(ku1)
	//Calc_dsigma(ksigma1)
	Calc_dsigmam(ksigma1, k1, escala)

	// stage 2
	Time = t0 + (1./2.)*Dt_si
	cuda.Madd2(m, m, k1, 1, (1./2.)*h) // m = m*1 + k1*h/2
	M.normalize()
	torqueFn(k2)
	cuda.Madd2(u, u, ku1, 1, (1./2.)*h0)
	//cuda.Madd2(r, r, kr1, 1, (1./2.)*h)
	cuda.Madd2(sigma, sigma, ksigma1, 1, (1./2.)*h0)
	Calc_du(ku2)
	//Calc_dsigma(ksigma2)
	Calc_dsigmam(ksigma2, k2, escala)
	//cuda.Madd2(kr2, kr2, u, 0, dt) // dx = dt * v0?

	// stage 3
	cuda.Madd2(m, m0, k2, 1, (1./2.)*h) // m = m0*1 + k2*1/2
	M.normalize()
	torqueFn(k3)
	cuda.Madd2(u, u0, ku2, 1, (1./2.)*h0)
	cuda.Madd2(sigma, sigma0, ksigma2, 1, (1./2.)*h0)
	Calc_du(ku3)
	//Calc_dsigma(ksigma3)
	Calc_dsigmam(ksigma3, k3, escala)

	// stage 4
	Time = t0 + Dt_si
	cuda.Madd2(m, m0, k3, 1, 1.*h) // m = m0*1 + k3*1
	M.normalize()
	torqueFn(k4)
	cuda.Madd2(u, u0, ku3, 1, 1.*h0)             // m = m0*1 + k3*1
	cuda.Madd2(sigma, sigma0, ksigma3, 1, 1.*h0) // m = m0*1 + k3*1
	Calc_du(ku4)
	//Calc_dsigma(ksigma4)
	Calc_dsigmam(ksigma4, k4, escala)

	err := cuda.MaxVecDiff(k1, k4) * float64(h)

	// adjust next time step
	if err < MaxErr || Dt_si <= MinDt || FixDt != 0 { // mindt check to avoid infinite loop
		// step OK
		// 4th order solution
		cuda.Madd5(m, m0, k1, k2, k3, k4, 1, (1./6.)*h, (1./3.)*h, (1./3.)*h, (1./6.)*h)
		M.normalize()
		NSteps++
		adaptDt(math.Pow(MaxErr/err, 1./4.))
		setLastErr(err)
		setMaxTorque(k4)
		cuda.Madd5(u, u0, ku1, ku2, ku3, ku4, 1, (1./6.)*h0, (1./3.)*h0, (1./3.)*h0, (1./6.)*h0)
		cuda.Madd5(sigma, sigma0, ksigma1, ksigma2, ksigma3, ksigma4, 1, (1./6.)*h0, (1./3.)*h0, (1./3.)*h0, (1./6.)*h0)
		cuda.Madd2(r, r, u, 1, h0) // x = x + dt * v
	} else {
		// undo bad step
		util.Assert(FixDt == 0)
		Time = t0
		data.Copy(m, m0)
		data.Copy(u, u0)
		data.Copy(sigma, sigma0)
		data.Copy(r, r0)
		NUndone++
		adaptDt(math.Pow(MaxErr/err, 1./5.))
	}
}

func (rk *MERK4s) Free() {}

// Elastic calculators
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

	cuda.AddMEField2(dst, M.Buffer(), Sigma.Buffer(),
		c11, c12, c44,
		b1, b2, ms, M.Mesh())
}

func AddStrainField(dst *data.Slice) {
	cuda.AddStrain(dst, R.Buffer(), M.Mesh())
}

func AddStrainField2(dst *data.Slice) {
	c11 := C11.MSlice()
	defer c11.Recycle()
	c12 := C12.MSlice()
	defer c12.Recycle()
	c44 := C44.MSlice()
	defer c44.Recycle()
	cuda.AddStrain2(dst, Sigma.Buffer(), c11, c12, c44, M.Mesh())
}

func AddStrainField3(dst *data.Slice) {
	c11 := C11.MSlice()
	defer c11.Recycle()
	c12 := C12.MSlice()
	defer c12.Recycle()
	c44 := C44.MSlice()
	defer c44.Recycle()
	cuda.AddStrain3(dst, Sigma.Buffer(), c11, c12, c44, M.Mesh())
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

// Kinetic Energy

func GetKineticEnergy(dst *data.Slice) {
	KineticEnergyDens(dst)
}

func KineticEnergyDens(dst *data.Slice) {
	rho := Rho.MSlice()
	defer rho.Recycle()
	cuda.KineticEnergy(dst, U.Buffer(), rho, M.Mesh())
}

func GetTotKineticEnergy() float64 {
	kinetic_energy := ValueOf(Edens_kin.Quantity)
	defer cuda.Recycle(kinetic_energy)
	return cellVolume() * float64(cuda.Sum(kinetic_energy))
}

// Elastic Energy

func GetElasticEnergy(dst *data.Slice) {
	ElasticEnergyDens(dst)
}

func ElasticEnergyDens(dst *data.Slice) {
	c1 := C11.MSlice()
	defer c1.Recycle()

	c2 := C12.MSlice()
	defer c2.Recycle()

	c3 := C44.MSlice()
	defer c3.Recycle()

	strain := cuda.Buffer(3, Mesh().Size())
	defer cuda.Recycle(strain)

	cuda.Zero(strain)
	AddStrainField(strain)

	cuda.ElasticEnergy(dst, strain, M.Mesh(), c1, c2, c3)
}

func GetTotElasticEnergy() float64 {
	el_energy := ValueOf(Edens_el.Quantity)
	defer cuda.Recycle(el_energy)
	return cellVolume() * float64(cuda.Sum(el_energy))
}

func GetMaxTorqueSigma() float64 {
	dsigma := cuda.Buffer(VECTOR, Sigma.Buffer().Size())
	defer cuda.Recycle(dsigma)
	dm := cuda.Buffer(VECTOR, M.Buffer().Size())
	defer cuda.Recycle(dm)
	torqueFn(dm)
	Calc_dsigmam(dsigma, dm, float32(GammaLL))
	return cuda.MaxVecNorm(dsigma)
}

func GetMaxTorqueSpeed() float64 {
	du := cuda.Buffer(VECTOR, U.Buffer().Size())
	defer cuda.Recycle(du)

	Calc_du(du)
	return cuda.MaxVecNorm(du)
}
