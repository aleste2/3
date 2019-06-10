package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/cuda/curand"
	"github.com/mumax/3/mag"
	//"github.com/mumax/3/util"
)

var (
	TCurie      = NewScalarParam("TCurie", "K", "Curie Temperature")
	Qext		= NewExcitation("Qext", "W/m3", "External Heating")

  // For Joule heating
	Langevin = 0
	JHThermalnoise = true
	RenormLLB   =false
	TSubsteps=3
	TOversteps=1
	TOverstepsCounter=1
	TempJH      LocalTemp   // Local temperature
	Kthermal    = NewScalarParam("Kthermal", "W/(m·K)", "Thermal conductivity")
	Cthermal    = NewScalarParam("Cthermal", "J/(Kg·K)", "Specific heat capacity")
	Resistivity = NewScalarParam("Resistivity", "Ohm·m", "Electric resistivity")
	Density     = NewScalarParam("Density", "Kg/m3", "Mass density")
	TSubs       = NewScalarParam("TSubs", "K", "Substrate Temperature")
	TauSubs     = NewScalarParam("TauSubs", "s", "Substrate difussion time")

	// For 3T Model
	Te      LocalTemp   // Electron temperature
	Tl      LocalTemp   // lattice temperature
	Ts      LocalTemp   // Spin temperature
	Kl    = NewScalarParam("Kl", "W/(m·K)", "Lattice thermal conductivity")
	Ke    = NewScalarParam("Ke", "W/(m·K)", "Electron thermal conductivity")
	Ks    = NewScalarParam("Ks", "W/(m·K)", "Spin thermal conductivity")
	Ce    = NewScalarParam("Ce", "J/(m3·K)", "Electron Heat capacity")
	Cl    = NewScalarParam("Cl", "J/(m3·K)", "Lattice Heat capacity")
	Cs    = NewScalarParam("Cs", "J/(m3·K)", "Spin Heat capacity")
	Gel    = NewScalarParam("Gel", "W/(m3·K)", "Transfer electron-lattice")
	Ges    = NewScalarParam("Ges", "W/(m3·K)", "Transfer electron-spin")
	Gls    = NewScalarParam("Gls", "W/(m3·K)", "Transfer lattice-spin")
	
	// For circular dichroism (only 3T model)
	CD	=NewVectorParam("CD", "", "Laser beam direction and Circular Dichroism magnitude")
)

func init() {
  // For JH (see at the end)
	DeclROnly("TempJH", AsScalarField(&TempJH), "Local Temperature (K)")
	TempJH.name="Local_Temperature"
	DeclFunc("RestartJH", StartJH, "Equals Temperature to substrate")
	DeclFunc("GetTemp", GetCell, "Gets cell temperature")

	// For 3T and 2T
	DeclFunc("Restart3T", Start3T, "Equals Temperatures to substrate")
	DeclFunc("Restart2T", Start2T, "Equals Temperatures to substrate")
	DeclROnly("Te", AsScalarField(&Te), "Electron Local Temperature (K)")
	Te.name="Local_Temperature_Electrons"
	DeclROnly("Tl", AsScalarField(&Tl), "Lattice Local Temperature (K)")
	Tl.name="Local_Temperature_Phonons"
	DeclROnly("Ts", AsScalarField(&Ts), "Spin Local Temperature (K)")
	Ts.name="Local_Temperature_Spins"
	DeclFunc("GetTe", GetTe, "Gets electron cell temperature")
	DeclFunc("GetTl", GetTl, "Gets lattice cell temperature")
	DeclFunc("GetTs", GetTs, "Gets spin cell temperature")

	DeclFunc("SetM", SetM, "Adjust m to temperature")
	DeclTVar("JHThermalnoise", &JHThermalnoise, "Enable/disable thermal noise")
	DeclTVar("Langevin", &Langevin, "Set M(T) to Langevin instead of Brillouin with J=1/2")
	DeclTVar("RenormLLB", &RenormLLB, "Enable/disable remormalize m in LLB")
	DeclVar("TSubsteps", &TSubsteps, "Number of substeps for Thermal equation")
	DeclVar("TOversteps", &TOversteps, "Number of oversteps for JH")
}

// LocalTemp definitions and Functions for JH

type LocalTemp struct {
	temp     *data.Slice      // noise buffer
	name string
}

// Update B_therm for LLB

func (b *thermField) LLBAddTo(dst *data.Slice) {
	if JHThermalnoise==true {
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

	if JHThermalnoise==false {
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
		if (solvertype<27) {
                               cuda.SetTemperature(dst.Comp(i), noise, k2_VgammaDt, ms, temp, alpha)
                               } else{
				if (solvertype==27){
			       		TempJH.update()
                               		cuda.SetTemperatureJH(dst.Comp(i), noise, k2_VgammaDt, ms, TempJH.temp, alpha)
				}else
				 if (solvertype==28){
			       		Te.update()
			       		Tl.update()
			       		Ts.update()
                               		cuda.SetTemperatureJH(dst.Comp(i), noise, k2_VgammaDt, ms, Ts.temp, alpha)
				}else
				 if (solvertype==29){
			       		Te.update()
			       		Tl.update()
                               		cuda.SetTemperatureJH(dst.Comp(i), noise, k2_VgammaDt, ms, Te.temp, alpha)
				}
                               }
	}

	b.step = NSteps
	b.dt = Dt_si
}



func StartJH() {
	TempJH.JHSetLocalTemp()
}

func Start3T() {
	Te.JHSetLocalTemp()
	Tl.JHSetLocalTemp()
	Ts.JHSetLocalTemp()
}

func Start2T() {
	Te.JHSetLocalTemp()
	Tl.JHSetLocalTemp()
}

func SetM() {
	TCurie := TCurie.MSlice()
	defer TCurie.Recycle()
	if solvertype==26 {
	Temp := Temp.MSlice()
	defer Temp.Recycle()
	cuda.InitmLLB(M.Buffer(),Temp,TCurie,Langevin)
	}
	if solvertype==27 {
	cuda.InitmLLBJH(M.Buffer(),TempJH.temp,TCurie,Langevin)
	}
	if solvertype==28 {
	cuda.InitmLLBJH(M.Buffer(),Ts.temp,TCurie,Langevin)
	}
	if solvertype==29 {
	cuda.InitmLLBJH(M.Buffer(),Te.temp,TCurie,Langevin)
	}
}

func (b *LocalTemp) JHSetLocalTemp() {
	b.update()
	TSubs := TSubs.MSlice()
	defer TSubs.Recycle()
	cuda.InitTemperatureJH(b.temp,TSubs)
}

func (b *LocalTemp) update() {
	if b.temp == nil {
		b.temp = cuda.NewSlice(b.NComp(), b.Mesh().Size())
		TSubs := TSubs.MSlice()
		defer TSubs.Recycle()
		cuda.InitTemperatureJH(b.temp,TSubs)
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
	return cuda.GetCell(b.temp, 0, ix, iy, iz)}

func GetCell(ix, iy, iz int) float64 {
	return float64(TempJH.GetCell(ix, iy, iz))
}

func GetTe(ix, iy, iz int) float64 {
	return float64(Te.GetCell(ix, iy, iz))
}

func GetTs(ix, iy, iz int) float64 {
	return float64(Ts.GetCell(ix, iy, iz))
}

func GetTl(ix, iy, iz int) float64 {
	return float64(Tl.GetCell(ix, iy, iz))
}
