package engine

// Elastic solver Core

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Epecemos usando tipos definidos y ya veremos luego. Inicialmente 2D
var (
	C11   = NewScalarExcitation("C11", "", "Elastic parameter C11")
	C12   = NewScalarExcitation("C12", "", "Elastic parameter C12")
	C44   = NewScalarExcitation("C44", "", "Elastic parameter C44")
	Rho   = NewScalarExcitation("rho", "Kg/m3", "Mass density")
	Force = NewScalarExcitation("force", "N/m3", "Force density")
	Eta   = NewScalarExcitation("eta", "", "Damping elastic")
	R     magnetization // displacement (m)
	U     magnetization // speed (m/s)
	Sigma magnetization // sigma (xx,yy xy) // 2D by now
)

func init() {
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

func Calc_du(dst *data.Slice) {

	eta := Eta.MSlice()
	defer eta.Recycle()
	rho := Rho.MSlice()
	defer rho.Recycle()
	force := Force.MSlice()
	defer force.Recycle()

	//c11 := C11.MSlice()
	//defer c11.Recycle()
	//c12 := C12.MSlice()
	//defer c12.Recycle()
	//c44 := C44.MSlice()
	//defer c44.Recycle()

	u := U.Buffer()
	sigma := Sigma.Buffer()

	cuda.CalcDU(dst, sigma, u, eta, rho, force, M.Mesh())
	NEvals++
}

func Calc_dsigma(dst *data.Slice) {

	//eta := Eta.MSlice()
	//defer eta.Recycle()
	//rho := Rho.MSlice()
	//defer rho.Recycle()
	//force := Force.MSlice()
	//defer force.Recycle()

	c11 := C11.MSlice()
	defer c11.Recycle()
	c12 := C12.MSlice()
	defer c12.Recycle()
	c44 := C44.MSlice()
	defer c44.Recycle()

	u := U.Buffer()
	//sigma := Sigma.Buffer()

	cuda.CalcDSigma(dst, u, c11, c12, c44, M.Mesh())
	NEvals++
}
