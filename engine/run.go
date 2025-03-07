package engine

import (
	"fmt"
	"math"
	"os"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Solver globals
var (
	Time                    float64                      // time in seconds
	alarm                   float64                      // alarm clock marks end time of run, dt adaptation must not cross it!
	pause                   = true                       // set pause at any time to stop running after the current step
	postStep                []func()                     // called on after every full time step
	Inject                           = make(chan func()) // injects code in between time steps. Used by web interface.
	Dt_si                   float64  = 1e-15             // time step = dt_si (seconds) *dt_mul, which should be nice float32
	MinDt, MaxDt            float64                      // minimum and maximum time step
	MaxErr                  float64  = 1e-5              // maximum error/step
	Headroom                float64  = 0.8               // solver headroom, (Gustafsson, 1992, Control of Error and Convergence in ODE Solvers)
	LastErr, PeakErr        float64                      // error of last step, highest error ever
	LastTorque              float64                      // maxTorque of last time step
	NSteps, NUndone, NEvals int                          // number of good steps, undone steps
	FixDt                   float64                      // fixed time step?
	stepper                 Stepper                      // generic step, can be EulerStep, HeunStep, etc
	solvertype              int
	// Added flags for for LLB and AF solvers
	LLBeq  = false
	LLBJHf = false
	LLB3Tf = false
	LLB2Tf = false
	AFf    = false
	OSC    = false
	Ato    = false
	MEf    = false
)

func init() {
	DeclFunc("Run", Run, "Run the simulation for a time in seconds")
	DeclFunc("Steps", Steps, "Run the simulation for a number of time steps")
	DeclFunc("RunWhile", RunWhile, "Run while condition function is true")
	DeclFunc("SetSolver", SetSolver, "Set solver type. 1:Euler, 2:Heun, 3:Bogaki-Shampine, 4: Runge-Kutta (RK45), 5: Dormand-Prince, 6: Fehlberg, -1: Backward Euler")
	DeclTVar("t", &Time, "Total simulated time (s)")
	DeclVar("step", &NSteps, "Total number of time steps taken")
	DeclVar("MinDt", &MinDt, "Minimum time step the solver can take (s)")
	DeclVar("MaxDt", &MaxDt, "Maximum time step the solver can take (s)")
	DeclVar("MaxErr", &MaxErr, "Maximum error per step the solver can tolerate (default = 1e-5)")
	DeclVar("Headroom", &Headroom, "Solver headroom (default = 0.8)")
	DeclVar("FixDt", &FixDt, "Set a fixed time step, 0 disables fixed step (which is the default)")
	DeclFunc("Exit", Exit, "Exit from the program")
	SetSolver(DORMANDPRINCE)
	_ = NewScalarValue("dt", "s", "Time Step", func() float64 { return Dt_si })
	_ = NewScalarValue("LastErr", "", "Error of last step", func() float64 { return LastErr })
	_ = NewScalarValue("PeakErr", "", "Overall maxium error per step", func() float64 { return PeakErr })
	_ = NewScalarValue("NEval", "", "Total number of torque evaluations", func() float64 { return float64(NEvals) })
}

// Time stepper like Euler, Heun, RK23
type Stepper interface {
	Step() // take time step using solver globals
	Free() // free resources, if any (e.g.: RK23 previous torque)
}

// Arguments for SetSolver
const (
	BACKWARD_EULER = -1
	EULER          = 1
	HEUN           = 2
	BOGAKISHAMPINE = 3
	RUNGEKUTTA     = 4
	DORMANDPRINCE  = 5
	FEHLBERG       = 6

	LLB   = 26
	LLBJH = 27
	LLB2T = 29

	//LLB2TOST = 30

	LLGJH = 50

	ANTIFERRORK4  = 100
	ANTIFERRORK23 = 101

	ANTIFERROLLBPRB = 127
	LLBAF2T         = 129
	LLBAFJH         = 130

	HEUNJHONLY = 207
	ONLY2T     = 208

	// Atomistic solvers
	ATOHEUN   = 150
	ATOHEUN2T = 151
	ATORK23   = 152

	// My magnetoelastics solvers
	ELASTIC  = 200
	MELASTIC = 201
)

func SetSolver(typ int) {
	// free previous solver, if any
	if stepper != nil {
		stepper.Free()
	}

	// Flags for temperature
	LLBeq = false
	LLBJHf = false
	LLB3Tf = false
	LLB2Tf = false
	AFf = false
	MEf = false

	switch typ {
	default:
		util.Fatalf("SetSolver: unknown solver type: %v", typ)
	case BACKWARD_EULER:
		stepper = new(BackwardEuler)
	case EULER:
		stepper = new(Euler)
	case HEUN:
		stepper = new(Heun)
	case BOGAKISHAMPINE:
		stepper = new(RK23)
	case RUNGEKUTTA:
		stepper = new(RK4)
	case DORMANDPRINCE:
		stepper = new(RK45DP)
	case FEHLBERG:
		stepper = new(RK56)
		//      LLB Solvers
	case LLB:
		//stepper = new(HeunLLB)
		stepper = new(HeunLLBFerroUnified)
		LLBeq = true
		LLB2Tf = false
	case LLBJH:
		//stepper = new(HeunLLBJH)
		stepper = new(HeunLLBFerroUnified)
		LLBeq = true
		LLBJHf = true
		LLB2Tf = false
	case LLB2T:
		//stepper = new(HeunLLB2T)
		stepper = new(HeunLLBFerroUnified)
		LLBeq = true
		LLB2Tf = true
		LLBJHf = false
	/*case LLB2TOST:
	stepper = new(HeunLLB2TOSC)
	LLBeq = true
	LLB2Tf = true
	OSC = true*/

	case LLGJH:
		stepper = new(LLGJHSolver)
		LLBJHf = true
		LLBeq = false
		LLB2Tf = false
		OSC = false

		// Antiferro solvers

	case ANTIFERRORK4:
		stepper = new(AntiferroRK4)
		AFf = true
	case ANTIFERRORK23:
		stepper = new(AntiferroRK23)
		AFf = true
	case ANTIFERROLLBPRB:
		//stepper = new(HeunAFLLBPRB054401)
		stepper = new(HeunLLBAFUnified)
		AFf = true
		LLBeq = true
	case LLBAF2T:
		//stepper = new(HeunLLBAF2T)
		stepper = new(HeunLLBAFUnified)
		LLBeq = true
		AFf = true
		LLB2Tf = true
		LLBJHf = false
	case LLBAFJH:
		//stepper = new(HeunLLB2T)
		stepper = new(HeunLLBAFUnified)
		LLBeq = true
		AFf = true
		LLB2Tf = false
		LLBJHf = true

		// Thermal only solvers
	case HEUNJHONLY:
		stepper = new(OnlyJH)
	case ONLY2T:
		stepper = new(Only2T)

		//      atomistic Solvers
	case ATOHEUN:
		stepper = new(HeunAto)
		Ato = true
	case ATOHEUN2T:
		stepper = new(HeunAto2T)
		LLB2Tf = true
		Ato = true
	case ATORK23:
		stepper = new(AtoRK23)
		Ato = true

		// Elastic solver
	case ELASTIC:
		//stepper = new(ElasticEuler)
		stepper = new(ElasticRK4)
	case MELASTIC:
		MEf = true
		stepper = new(MERK4s)

	}

	solvertype = typ
}

// write torque to dst and increment NEvals
func torqueFn(dst *data.Slice) {
	SetTorque(dst)
	NEvals++
}

/*
////////////////////////////////////   Added for LLB

func torqueFnLLB(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {
	SetTorqueLLB(dst, hth1, hth2)
	NEvals++
}

////////////////////////////////////   Added for LLBJH

func torqueFnLLBJH(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {
	SetTorqueLLBJH(dst, hth1, hth2)
	NEvals++
}

////////////////////////////////////   Added for LLB 3T

func torqueFnLLB3T(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {
	SetTorqueLLB3T(dst, hth1, hth2)
	NEvals++
}

////////////////////////////////////   Added for LLB 2T

func torqueFnLLB2T(dst *data.Slice, hth1 *data.Slice, hth2 *data.Slice) {
	SetTorqueLLB2T(dst, hth1, hth2)
	NEvals++
}

*/

// returns number of torque evaluations
func getNEval() int {
	return NEvals
}

// update lastErr and peakErr
func setLastErr(err float64) {
	LastErr = err
	if err > PeakErr {
		PeakErr = err
	}
}

func setMaxTorque(τ *data.Slice) {
	LastTorque = cuda.MaxVecNorm(τ)
}

// adapt time step: dt *= corr, but limited to sensible values.
func adaptDt(corr float64) {
	if FixDt != 0 {
		Dt_si = FixDt
		return
	}

	// corner case triggered by err = 0: just keep time step.
	// see test/regression017.mx3
	if math.IsNaN(corr) {
		corr = 1
	}

	util.AssertMsg(corr != 0, "Time step too small, check if parameters are sensible")
	corr *= Headroom
	if corr > 2 {
		corr = 2
	}
	if corr < 1./2. {
		corr = 1. / 2.
	}
	Dt_si *= corr
	if MinDt != 0 && Dt_si < MinDt {
		Dt_si = MinDt
	}
	if MaxDt != 0 && Dt_si > MaxDt {
		Dt_si = MaxDt
	}
	if Dt_si == 0 {
		util.Fatal("time step too small")
	}

	// do not cross alarm time
	if Time < alarm && Time+Dt_si > alarm {
		Dt_si = alarm - Time
	}

	util.AssertMsg(Dt_si > 0, fmt.Sprint("Time step too small: ", Dt_si))
}

// Run the simulation for a number of seconds.
func Run(seconds float64) {
	stop := Time + seconds
	alarm = stop // don't have dt adapt to go over alarm
	RunWhile(func() bool { return Time < stop })
}

// Run the simulation for a number of steps.
func Steps(n int) {
	stop := NSteps + n
	RunWhile(func() bool { return NSteps < stop })
}

// Runs as long as condition returns true, saves output.
func RunWhile(condition func() bool) {
	SanityCheck()
	pause = false // may be set by <-Inject
	const output = true
	stepper.Free() // start from a clean state
	runWhile(condition, output)
	pause = true
}

func runWhile(condition func() bool, output bool) {
	DoOutput() // allow t=0 output
	for condition() && !pause {
		select {
		default:
			step(output)
		// accept tasks form Inject channel
		case f := <-Inject:
			f()
		}
	}
}

// Runs as long as browser is connected to gui.
func RunInteractive() {
	gui_.RunInteractive()
}

// take one time step
func step(output bool) {
	stepper.Step()
	for _, f := range postStep {
		f()
	}
	if output {
		DoOutput()
	}
}

// Register function f to be called after every time step.
// Typically used, e.g., to manipulate the magnetization.
func PostStep(f func()) {
	postStep = append(postStep, f)
}

// inject code into engine and wait for it to complete.
func InjectAndWait(task func()) {
	ready := make(chan int)
	Inject <- func() { task(); ready <- 1 }
	<-ready
}

func SanityCheck() {
	if Msat.isZero() {
		util.Log("Note: Msat = 0")
	}
	if Aex.isZero() {
		util.Log("Note: Aex = 0")
	}
}

func Exit() {
	Close()
	os.Exit(0)
}
