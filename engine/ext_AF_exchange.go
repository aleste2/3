package engine

// Exchange interaction (Heisenberg + Dzyaloshinskii-Moriya) for AF implementation
// See also cuda/exchange.cu and cuda/dmi.cu

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Bex    = NewScalarParam("Bex", "J/m", "Exchange stiffness interlattice same cell")

	Aexll   = NewScalarParam("Aexll", "J/m", "Exchange stiffness interlattice different cell", &lexll)

	Aex1   = NewScalarParam("Aex1", "J/m", "Exchange stiffness lattice 1", &lex21)
	Dind1  = NewScalarParam("Dind1", "J/m2", "Interfacial Dzyaloshinskii-Moriya strength lattice 1", &din21)
	Dbulk1 = NewScalarParam("Dbulk1", "J/m2", "Bulk Dzyaloshinskii-Moriya strength lattice 1", &dbulk21)
	Aex2   = NewScalarParam("Aex2", "J/m", "Exchange stiffness lattice 1", &lex22)
	Dind2  = NewScalarParam("Dind2", "J/m2", "Interfacial Dzyaloshinskii-Moriya strength lattice 2", &din22)
	Dbulk2 = NewScalarParam("Dbulk2", "J/m2", "Bulk Dzyaloshinskii-Moriya strength lattice 2", &dbulk22)

	lexll       aexchParam // inter-cell exchange in 1e18 * Aex 

	lex21       aexchParam // inter-cell exchange in 1e18 * Aex / Msat
	din21       dexchParam // inter-cell interfacial DMI in 1e9 * Dex / Msat
	dbulk21     dexchParam // inter-cell bulk DMI in 1e9 * Dex / Msat
	lex22       aexchParam // inter-cell exchange in 1e18 * Aex / Msat
	din22       dexchParam // inter-cell interfacial DMI in 1e9 * Dex / Msat
	dbulk22     dexchParam // inter-cell bulk DMI in 1e9 * Dex / Msat
	DindCoupling1 = NewScalarField("DindCoupling1", "arb.", "Average DMI coupling with neighbors", dindDecodeAF1)
	DindCoupling2 = NewScalarField("DindCoupling2", "arb.", "Average DMI coupling with neighbors", dindDecodeAF2)
)

func init() {
	DeclFunc("ext_ScaleExchangeAF1", ScaleInterExchangeAF1, "Re-scales exchange coupling between two regions.")
	DeclFunc("ext_ScaleExchangeAF2", ScaleInterExchangeAF2, "Re-scales exchange coupling between two regions.")

	lex21.init()
	din21.init(Dind1)
	dbulk21.init(Dbulk1)
	lex22.init()
	din22.init(Dind2)
	dbulk22.init(Dbulk2)
	lexll.init()
}

// Adds the current exchange AFfield to dst
func AddExchangeFieldAF(dst1,dst2 *data.Slice) {

	//Sublattice 1
	inter := !Dind1.isZero()
	bulk := !Dbulk1.isZero()
	switch {
	case !inter && !bulk:
		cuda.AddExchange(dst1, M1.Buffer(), lex21.GpuAF1(), regions.Gpu(), M.Mesh())
	case inter && !bulk:
		cuda.AddDMI(dst1, M1.Buffer(), lex21.GpuAF1(), din21.GpuAF1(), regions.Gpu(), M.Mesh()) // dmi+exchange
	case bulk && !inter:
		util.AssertMsg(allowUnsafe || (Msat1.IsUniform() && Aex1.IsUniform() && Dbulk1.IsUniform()), "DMI: Msat, Aex, Dex must be uniform")
		cuda.AddDMIBulk(dst1, M1.Buffer(), lex21.GpuAF1(), dbulk21.GpuAF1(), regions.Gpu(), M.Mesh()) // dmi+exchange
	case inter && bulk:
		util.Fatal("Cannot have induced and interfacial DMI at the same time")
	}

	//Sublattice 2
	inter = !Dind2.isZero()
	bulk = !Dbulk2.isZero()
	switch {
	case !inter && !bulk:
		cuda.AddExchange(dst2, M2.Buffer(), lex22.GpuAF2(), regions.Gpu(), M.Mesh())
	case inter && !bulk:
		cuda.AddDMI(dst2, M2.Buffer(), lex22.GpuAF2(), din22.GpuAF2(), regions.Gpu(), M.Mesh()) // dmi+exchange
	case bulk && !inter:
		util.AssertMsg(allowUnsafe || (Msat2.IsUniform() && Aex2.IsUniform() && Dbulk2.IsUniform()), "DMI: Msat, Aex, Dex must be uniform")
		cuda.AddDMIBulk(dst2, M2.Buffer(), lex22.GpuAF2(), dbulk22.GpuAF2(), regions.Gpu(), M.Mesh()) // dmi+exchange
	case inter && bulk:
		util.Fatal("Cannot have induced and interfacial DMI at the same time")
	}

	
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	bex := Bex.MSlice()
	defer bex.Recycle()
	cuda.AddExchangeAFCell(dst1,dst2,M1.Buffer(),M2.Buffer(),ms1,ms2,bex)
	cuda.AddExchangeAFll(dst1,dst2,M1.Buffer(),M2.Buffer(),ms1,ms2,lexll.Gpull(), regions.Gpu(), M.Mesh())
}


// Set dst to the average exchange coupling per cell (average of lex2 with all neighbors).
func exchangeDecodeAF1(dst *data.Slice) {
	cuda.ExchangeDecode(dst, lex21.Gpu(), regions.Gpu(), M.Mesh())
}
func exchangeDecodeAF2(dst *data.Slice) {
	cuda.ExchangeDecode(dst, lex22.Gpu(), regions.Gpu(), M.Mesh())
}

func exchangeDecodell(dst *data.Slice) {
	cuda.ExchangeDecode(dst, lexll.Gpu(), regions.Gpu(), M.Mesh())
}


// Set dst to the average dmi coupling per cell (average of din2 with all neighbors).
func dindDecodeAF1(dst *data.Slice) {
	cuda.ExchangeDecode(dst, din21.Gpu(), regions.Gpu(), M.Mesh())
}
func dindDecodeAF2(dst *data.Slice) {
	cuda.ExchangeDecode(dst, din22.Gpu(), regions.Gpu(), M.Mesh())
}

// Scales the heisenberg exchange interaction between region1 and 2.
// Scale = 1 means the harmonic mean over the regions of Aex/Msat.
func ScaleInterExchangeAF1(region1, region2 int, scale float64) {
	lex21.scale[symmidx(region1, region2)] = float32(scale)
	lex21.invalidate()
}
func ScaleInterExchangeAF2(region1, region2 int, scale float64) {
	lex22.scale[symmidx(region1, region2)] = float32(scale)
	lex22.invalidate()
}

func ScaleInterExchangell(region1, region2 int, scale float64) {
	lexll.scale[symmidx(region1, region2)] = float32(scale)
	lexll.invalidate()
}



// Scales the DMI interaction between region 1 and 2.
func ScaleInterDindAF1(region1, region2 int, scale float64) {
	din21.scale[symmidx(region1, region2)] = float32(scale)
	din21.invalidate()
}
func ScaleInterDindAF2(region1, region2 int, scale float64) {
	din22.scale[symmidx(region1, region2)] = float32(scale)
	din22.invalidate()
}

// Sets the DMI interaction between region 1 and 2.
func InterDindAF1(region1, region2 int, value float64) {
	din21.scale[symmidx(region1, region2)] = float32(0.)
	din21.interdmi[symmidx(region1, region2)] = float32(value)
	din21.invalidate()
}
func InterDindAF2(region1, region2 int, value float64) {
	din22.scale[symmidx(region1, region2)] = float32(0.)
	din22.interdmi[symmidx(region1, region2)] = float32(value)
	din22.invalidate()
}

func (p *aexchParam) updateAF1() {
	if !p.cpu_ok {
		msat := Msat1.cpuLUT()
		aex := Aex1.cpuLUT()

		for i := 0; i < NREGION; i++ {
			lexi := 1e18 * safediv(aex[0][i], msat[0][i])
			for j := i; j < NREGION; j++ {
				lexj := 1e18 * safediv(aex[0][j], msat[0][j])
				I := symmidx(i, j)
				p.lut[I] = p.scale[I] * 2 / (1/lexi + 1/lexj)
			}
		}
		p.gpu_ok = false
		p.cpu_ok = true
	}
}
func (p *aexchParam) updateAF2() {
	if !p.cpu_ok {
		msat := Msat2.cpuLUT()
		aex := Aex2.cpuLUT()

		for i := 0; i < NREGION; i++ {
			lexi := 1e18 * safediv(aex[0][i], msat[0][i])
			for j := i; j < NREGION; j++ {
				lexj := 1e18 * safediv(aex[0][j], msat[0][j])
				I := symmidx(i, j)
				p.lut[I] = p.scale[I] * 2 / (1/lexi + 1/lexj)
			}
		}
		p.gpu_ok = false
		p.cpu_ok = true
	}
}
func (p *aexchParam) updatell() {
	if !p.cpu_ok {
		msat := float32(1.0)
		aex := Aexll.cpuLUT()

		for i := 0; i < NREGION; i++ {
			lexi := 1e18 * safediv(aex[0][i], msat)
			for j := i; j < NREGION; j++ {
				lexj := 1e18 * safediv(aex[0][j], msat)
				I := symmidx(i, j)
				p.lut[I] = p.scale[I] * 2 / (1/lexi + 1/lexj)
			}
		}
		p.gpu_ok = false
		p.cpu_ok = true
	}
}

func (p *dexchParam) updateAF1() {
	if !p.cpu_ok {
		msat := Msat1.cpuLUT()
		dex := p.parent.cpuLUT()
		for i := 0; i < NREGION; i++ {
			dexi := 1e9 * safediv(dex[0][i], msat[0][i])
			for j := i; j < NREGION; j++ {
				dexj := 1e9 * safediv(dex[0][j], msat[0][j])
				I := symmidx(i, j)
				interdmi := 1e9 * safediv(p.interdmi[I], msat[0][i])
				p.lut[I] = p.scale[I]*2/(1/dexi+1/dexj) + interdmi
			}
		}
		p.gpu_ok = false
		p.cpu_ok = true
	}
}
func (p *dexchParam) updateAF2() {
	if !p.cpu_ok {
		msat := Msat2.cpuLUT()
		dex := p.parent.cpuLUT()
		for i := 0; i < NREGION; i++ {
			dexi := 1e9 * safediv(dex[0][i], msat[0][i])
			for j := i; j < NREGION; j++ {
				dexj := 1e9 * safediv(dex[0][j], msat[0][j])
				I := symmidx(i, j)
				interdmi := 1e9 * safediv(p.interdmi[I], msat[0][i])
				p.lut[I] = p.scale[I]*2/(1/dexi+1/dexj) + interdmi
			}
		}
		p.gpu_ok = false
		p.cpu_ok = true
	}
}

// Get a GPU mirror of the look-up table.
// Copies to GPU first only if needed.
func (p *dexchParam) GpuAF1() cuda.SymmLUT {
	p.updateAF1()
	if !p.gpu_ok {
		p.upload()
	}
	return p.gpu
}
func (p *dexchParam) GpuAF2() cuda.SymmLUT {
	p.updateAF2()
	if !p.gpu_ok {
		p.upload()
	}
	return p.gpu
}

func (p *aexchParam) GpuAF1() cuda.SymmLUT {
	p.updateAF1()
	if !p.gpu_ok {
		p.upload()
	}
	return p.gpu
	// TODO: dedup
}
func (p *aexchParam) GpuAF2() cuda.SymmLUT {
	p.updateAF2()
	if !p.gpu_ok {
		p.upload()
	}
	return p.gpu
	// TODO: dedup
}
func (p *aexchParam) Gpull() cuda.SymmLUT {
	p.updatell()
	if !p.gpu_ok {
		p.upload()
	}
	return p.gpu
	// TODO: dedup
}
