package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/data"
	"github.com/mumax/3/mag"
	//"github.com/mumax/3/util"
	"math"
)

var (
	Qext = NewExcitation("Qext", "W/m3", "External Heating")

	// For Joule heating
	Langevin               = 0
	JHThermalnoise         = true
	ScaleNoiseLLB  float64 = 1.0 // reduce noise in LLB
	TSubsteps              = 3
	//TOversteps                  = 1
	//TOverstepsCounter           = 0
	//flagOST                     = 0
	//TempJH            LocalTemp // Local temperature
	//	Kthermal          = NewScalarParam("Kthermal", "W/(m·K)", "Thermal conductivity")
	Kthermal    = NewScalarParam("Kthermal", "J/(m3·K)", "Electron Heat capacity", &kth)
	Cthermal    = NewScalarParam("Cthermal", "J/(Kg·K)", "Specific heat capacity")
	Resistivity = NewScalarParam("Resistivity", "Ohm·m", "Electric resistivity")
	Density     = NewScalarParam("Density", "Kg/m3", "Mass density")
	TSubs       = NewScalarParam("TSubs", "K", "Substrate Temperature")
	TauSubs     = NewScalarParam("TauSubs", "s", "Substrate difussion time")

	// For 2T Model
	Te LocalTemp // Electron temperature
	Tl LocalTemp // lattice temperature
	// Kl  = NewScalarParam("Kl", "W/(m·K)", "Lattice thermal conductivity")
	// Ke  = NewScalarParam("Ke", "W/(m·K)", "Electron thermal conductivity")
	//Ks  = NewScalarParam("Ks", "W/(m·K)", "Spin thermal conductivity")
	Ce  = NewScalarParam("Ce", "J/(m3·K)", "Electron Heat capacity")
	Cl  = NewScalarParam("Cl", "J/(m3·K)", "Lattice Heat capacity")
	Gel = NewScalarParam("Gel", "W/(m3·K)", "Transfer electron-lattice")

	// Trial Ke Ka type exchange for thermal resistance and different materials
	Kl  = NewScalarParam("Kl", "W/(m·K)", "Lattice thermal conductivity", &kll)
	Ke  = NewScalarParam("Ke", "W/(m·K)", "Electron thermal conductivity", &kel)
	kel exchParam // electron themal contuctivity
	kll exchParam // lattice themal contuctivity
	kth exchParam // lattice themal contuctivity

	// For circular dichroism (only 3T model)
	CD = NewVectorParam("CD", "", "Laser beam direction and Circular Dichroism magnitude")
)

func init() {
	// For JH (see at the end)
	//DeclROnly("TempJH", AsScalarField(&TempJH), "Local Temperature (K)")
	//TempJH.name = "Local_Temperature"
	DeclFunc("RestartJH", StartJH, "Equals Temperature to substrate")
	//DeclFunc("GetTemp", GetCell, "Gets cell temperature")

	// For 2T
	DeclFunc("Restart2T", Start2T, "Equals Temperatures to substrate")
	DeclROnly("Te", AsScalarField(&Te), "Electron Local Temperature (K)")
	Te.name = "Local_Temperature_Electrons"
	DeclROnly("Tl", AsScalarField(&Tl), "Lattice Local Temperature (K)")
	Tl.name = "Local_Temperature_Phonons"
	DeclFunc("GetTe", GetTe, "Gets electron cell temperature")
	DeclFunc("GetTl", GetTl, "Gets lattice cell temperature")
	DeclFunc("SetTl", SetTlToTe, "Set Tl to Te")

	DeclFunc("RadialMask", RadialMask, "Gaussian mask")

	//DeclFunc("SetM", SetM, "Adjust m to temperature")
	DeclTVar("JHThermalnoise", &JHThermalnoise, "Enable/disable thermal noise")
	//DeclTVar("Langevin", &Langevin, "Set M(T) to Langevin instead of Brillouin with J=1/2")
	//DeclTVar("RenormLLB", &RenormLLB, "Enable/disable remormalize m in LLB")
	DeclVar("TSubsteps", &TSubsteps, "Number of substeps for Thermal equation")
	//DeclVar("TOversteps", &TOversteps, "Number of oversteps for JH")
	DeclVar("ScaleNoiseLLB", &ScaleNoiseLLB, "Thermal noise scale")

	// For new thermal difussion
	kel.init(Ke)
	kll.init(Kl)
	kth.init(Kthermal)
	DeclFunc("ext_InterExchangeKe", InterExchangeKe, "Sets electron thermal difussion between two regions.")
	DeclFunc("ext_InterExchangeKl", InterExchangeKl, "Sets lattice thermal difussion between two regions.")
}

// Sets electron thermal difussion between two regions
func InterExchangeKe(region1, region2 int, value float64) {
	kel.setInter(region1, region2, value)
	kel.update()
}

// Sets lattice thermal difussion between two regions
func InterExchangeKl(region1, region2 int, value float64) {
	kll.setInter(region1, region2, value)
	kll.update()
}

// LocalTemp definitions and Functions for JH

type LocalTemp struct {
	temp *data.Slice // noise buffer
	name string
}

// Update B_therm for LLB

func (b *thermField) LLBAddTo(dst *data.Slice) {
	if JHThermalnoise == true {
		b.LLBupdate()
		cuda.Add(dst, dst, b.noise)
	}
}

func (b *thermField) LLBupdate() {
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

	if JHThermalnoise == false {
		cuda.Memset(b.noise, 0, 0, 0)
		b.step = NSteps
		b.dt = Dt_si
		return
	}

	// keep constant during time step
	if NSteps == b.step && Dt_si == b.dt && solvertype < 6 {
		//if NSteps == b.step && Dt_si == b.dt {
		return
	}

	if FixDt == 0 {
		//util.Fatal("Finite temperature requires fixed time step. Set FixDt != 0.")
	}

	N := Mesh().NCell()
	k2_VgammaDt := 2 * mag.Kb / (GammaLL * cellVolume() * Dt_si)
	noise := cuda.Buffer(1, Mesh().Size())
	defer cuda.Recycle(noise)

	const mean = 0
	const stddev = 1
	dst := b.noise
	ms := Msat.MSlice()
	defer ms.Recycle()
	temp := Temp.MSlice()
	defer temp.Recycle()
	alpha := Alpha.MSlice()
	defer alpha.Recycle()
	for i := 0; i < 3; i++ {
		b.generator.GenerateNormal(uintptr(noise.DevPtr(0)), int64(N), mean, stddev)
		if LLBJHf == true {
			Te.update()
			cuda.SetTemperatureJH(dst.Comp(i), noise, k2_VgammaDt, ms, Te.temp, alpha, ScaleNoiseLLB)
		}
		if LLB2Tf == true {
			Te.update()
			Tl.update()
			cuda.SetTemperatureJH(dst.Comp(i), noise, k2_VgammaDt, ms, Te.temp, alpha, ScaleNoiseLLB)
		}
	}

	b.step = NSteps
	b.dt = Dt_si
}

func StartJH() {
	//TempJH.JHSetLocalTemp()
	Te.JHSetLocalTemp()
}

func Start2T() {
	Te.JHSetLocalTemp()
	Tl.JHSetLocalTemp()
}

func SetTlToTe() {
	data.Copy(Tl.temp, Te.temp)
}

func (b *LocalTemp) JHSetLocalTemp() {
	b.update()
	TSubs := TSubs.MSlice()
	defer TSubs.Recycle()
	cuda.InitTemperatureJH(b.temp, TSubs)
}

func (b *LocalTemp) update() {
	if b.temp == nil {
		b.temp = cuda.NewSlice(b.NComp(), b.Mesh().Size())
		TSubs := TSubs.MSlice()
		defer TSubs.Recycle()
		cuda.InitTemperatureJH(b.temp, TSubs)
	}
}

func (b *LocalTemp) Mesh() *data.Mesh       { return Mesh() }
func (b *LocalTemp) NComp() int             { return 1 }
func (b *LocalTemp) Name() string           { return b.name }
func (b *LocalTemp) Unit() string           { return "K" }
func (b *LocalTemp) average() []float64     { return qAverageUniverse(b) }
func (b *LocalTemp) EvalTo(dst *data.Slice) { EvalTo(b, dst) }
func (b *LocalTemp) Slice() (*data.Slice, bool) {
	b.update()
	return b.temp, false
}

func (b *LocalTemp) GetCell(ix, iy, iz int) float32 {
	return cuda.GetCell(b.temp, 0, ix, iy, iz)
}

//func GetCell(ix, iy, iz int) float64 {
//	return float64(TempJH.GetCell(ix, iy, iz))
//}

func GetTe(ix, iy, iz int) float64 {
	return float64(Te.GetCell(ix, iy, iz))
}

func GetTl(ix, iy, iz int) float64 {
	return float64(Tl.GetCell(ix, iy, iz))
}

func RadialMask(mascara *data.Slice, xc, yc, r0 float64, Nx1,Nx2, Ny1,Ny2,Nz1,Nz2 int) {
	for i := Nx1; i < Nx2; i++ {
		for j := Ny1; j < Ny2; j++ {
			for k := Nz1; k < Nz2; k++ {
			  r := Index2Coord(i, j, 0)
			  x := r.X()
			  y := r.Y()
			  dr := math.Sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc))            // Distancia al centro del Laser
			  LS := math.Exp(-4.0 * math.Log(2.0) * math.Pow(dr/r0, 2)) // Gaussian Laser Spot (circular)
			  mascara.SetVector(i, j, k, Vector(LS, 0, 0))              // For Q_ext
		  }
		}
	}

}
