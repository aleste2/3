package engine

// Relax tries to find the minimum energy state.

import (
	"github.com/mumax/3/cuda"
	"math"
)

func init() {
	DeclFunc("RelaxAF", RelaxAF, "Try to minimize the total energy in antiferro")
}


func RelaxAF() {
	SanityCheck()
	pause = false

	// Save the settings we are changing...
	prevType := solvertype
	prevErr := MaxErr
	prevFixDt := FixDt
	prevPrecess := Precess

	// ...to restore them later
	defer func() {
		SetSolver(prevType)
		MaxErr = prevErr
		FixDt = prevFixDt
		Precess = prevPrecess
		relaxing = false
		//	Temp.upd_reg = prevTemp
		//	Temp.invalidate()
		//	Temp.update()
	}()

	// Set good solver for relax
	SetSolver(ANTIFERRORK23)   // to do ANTIFERRORK23
	FixDt = 0
	Precess = false
	relaxing = true

	// Minimize energy: take steps as long as energy goes down.
	// This stops when energy reaches the numerical noise floor.
	const N = 3 // evaluate energy (expensive) every N steps
	relaxSteps(N)
	E0 := GetTotalEnergy()
	relaxSteps(N)
	E1 := GetTotalEnergy()
	for E1 < E0 && !pause {
		relaxSteps(N)
		E0, E1 = E1, GetTotalEnergy()
	}

	// Now we are already close to equilibrium, but energy is too noisy to be used any further.
	// So now we minimize the total torque which is less noisy and does not have to cross any
	// bumps once we are close to equilibrium.
	solver := stepper.(*AntiferroRK23)  // To do *AntiferroRK23
	defer stepper.Free() // purge previous rk.k1 because FSAL will be dead wrong.
	avgTorque := func() float32 {
		return cuda.Dot(solver.k11, solver.k11)
	}
	var T0, T1 float32 = 0, avgTorque()

	// Step as long as torque goes down. Then increase the accuracy and step more.
	for MaxErr > 1e-9 && !pause {
		MaxErr /= math.Sqrt2
		relaxSteps(N) // TODO: Play with other values
		T0, T1 = T1, avgTorque()
		for T1 < T0 && !pause {
			relaxSteps(N) // TODO: Play with other values
			T0, T1 = T1, avgTorque()
		}
	}

	pause = true
}
