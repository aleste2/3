package engine

// Exchange interaction (Heisenberg + Dzyaloshinskii-Moriya) for AF implementation
// See also cuda/exchange.cu and cuda/dmi.cu

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var (
	Bex12 = NewScalarParam("Bex12", "J/m", "Exchange stiffness interlattice same cell 1-2")
	Bex21 = NewScalarParam("Bex21", "J/m", "Exchange stiffness interlattice same cell 2-1")

	Aexll = NewScalarParam("Aexll", "J/m", "Exchange stiffness interlattice different cell", &lexll)

	Aex1   = NewScalarParam("Aex1", "J/m", "Exchange stiffness lattice 1", &lex21)
	Dind1  = NewScalarParam("Dind1", "J/m2", "Interfacial Dzyaloshinskii-Moriya strength lattice 1", &din21)
	Dbulk1 = NewScalarParam("Dbulk1", "J/m2", "Bulk Dzyaloshinskii-Moriya strength lattice 1", &dbulk21)

	Aex2   = NewScalarParam("Aex2", "J/m", "Exchange stiffness lattice 2", &lex22)
	Dind2  = NewScalarParam("Dind2", "J/m2", "Interfacial Dzyaloshinskii-Moriya strength lattice 2", &din22)
	Dbulk2 = NewScalarParam("Dbulk2", "J/m2", "Bulk Dzyaloshinskii-Moriya strength lattice 2", &dbulk22)

	lexll exchParam // inter-cell exchange

	lex21   exchParam // inter-cell exchange
	din21   exchParam // inter-cell interfacial DMI
	dbulk21 exchParam // inter-cell bulk DMI

	lex22   exchParam // inter-cell exchange
	din22   exchParam // inter-cell interfacial DMI
	dbulk22 exchParam // inter-cell bulk DMI

	// Trial Exchange Fields
	B_exch1 = NewVectorField("B_exch1", "T", "Exchange field Subnet 1", AddExchangeField1)
	B_exch2 = NewVectorField("B_exch2", "T", "Exchange field Subnet 2", AddExchangeField2)
)

func init() {
	lex21.init(Aex1)
	din21.init(Dind1)
	dbulk21.init(Dbulk1)

	lex22.init(Aex2)
	din22.init(Dind2)
	dbulk22.init(Dbulk2)

	lexll.init(Aexll)

	DeclFunc("ext_InterExchangeAex1", InterExchangeAex1, "Sets exchange coupling between two regions.")
	DeclFunc("ext_InterExchangeAex2", InterExchangeAex2, "Sets exchange coupling between two regions.")
	DeclFunc("ext_InterExchangeAexll", InterExchangeAexll, "Sets exchange coupling between two regions.")
	DeclFunc("ext_ScaleExchangeAex1", ScaleInterExchangeAex1, "Re-scales exchange coupling between two regions.")
	DeclFunc("ext_ScaleExchangeAex2", ScaleInterExchangeAex2, "Re-scales exchange coupling between two regions.")
	DeclFunc("ext_ScaleExchangeAexll", ScaleInterExchangeAexll, "Re-scales exchange coupling between two regions.")
}

// Adds the current exchange AFfield to dst
func AddExchangeFieldAF(dst1, dst2 *data.Slice) {

	// Write eveything new to use both subnets
	//Sublattice 1
	inter1 := !Dind1.isZero()
	bulk1 := !Dbulk1.isZero()
	ms1 := Msat1.MSlice()
	defer ms1.Recycle()
	//Sublattice 2
	//inter2 := !Dind1.isZero()
	//bulk2 := !Dbulk1.isZero()
	ms2 := Msat2.MSlice()
	defer ms2.Recycle()
	bex12 := Bex12.MSlice()
	defer bex12.Recycle()
	bex21 := Bex21.MSlice()
	defer bex21.Recycle()

	// Asumo que manda la red 1 si se ponen parámetros cuzados bulk dmi se va todo al carajo by now

	switch {
	case !inter1 && !bulk1:
		cuda.AddExchange(dst1, M1.Buffer(), lex21.Gpu(), ms1, regions.Gpu(), M.Mesh())
		cuda.AddExchange(dst2, M2.Buffer(), lex22.Gpu(), ms2, regions.Gpu(), M.Mesh())
		cuda.AddExchangeAFCell(dst1, dst2, M1.Buffer(), M2.Buffer(), ms1, ms2, bex12, bex21)
		cuda.AddExchangeAFll(dst1, dst2, M1.Buffer(), M2.Buffer(), ms1, ms2, lexll.Gpu(), regions.Gpu(), M.Mesh())
	case inter1 && !bulk1: // Ahora es un 2x1 con todos los exchanges
		cuda.AddExchangeAFCell(dst1, dst2, M1.Buffer(), M2.Buffer(), ms1, ms2, bex12, bex21)                                                                                          //Linea añadida 29/07/2024
		cuda.AddDMIAF(dst1, dst2, M1.Buffer(), M2.Buffer(), lex21.Gpu(), din21.Gpu(), lex22.Gpu(), din22.Gpu(), lexll.Gpu(), ms1, ms2, bex12, bex21, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
	case bulk1 && !inter1:
		cuda.AddDMIBulk(dst1, M1.Buffer(), lex21.Gpu(), dbulk21.Gpu(), ms1, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		cuda.AddDMIBulk(dst2, M2.Buffer(), lex22.Gpu(), dbulk22.Gpu(), ms2, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		cuda.AddExchangeAFCell(dst1, dst2, M1.Buffer(), M2.Buffer(), ms1, ms2, bex12, bex21)
		cuda.AddExchangeAFll(dst1, dst2, M1.Buffer(), M2.Buffer(), ms1, ms2, lexll.Gpu(), regions.Gpu(), M.Mesh())
	case inter1 && bulk1:
		util.Fatal("Cannot have induced and interfacial DMI at the same time")

	}

	/*
		//Sublattice 1
		inter := !Dind1.isZero()
		bulk := !Dbulk1.isZero()
		ms1 := Msat1.MSlice()
		defer ms1.Recycle()
		switch {
		case !inter && !bulk:
			cuda.AddExchange(dst1, M1.Buffer(), lex21.Gpu(), ms1, regions.Gpu(), M.Mesh())
		case inter && !bulk:
			//		cuda.AddDMI(dst1, M1.Buffer(), lex21.Gpu(), din21.Gpu(), ms1, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
			cuda.AddDMIAF(dst1, M1.Buffer(), M2.Buffer(), lex21.Gpu(), din21.Gpu(), lex22.Gpu(), din22.Gpu(), lexll.Gpu(), ms1, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		case bulk && !inter:
			cuda.AddDMIBulk(dst1, M1.Buffer(), lex21.Gpu(), dbulk21.Gpu(), ms1, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		case inter && bulk:
			util.Fatal("Cannot have induced and interfacial DMI at the same time")
		}

		//Sublattice 2
		inter = !Dind2.isZero()
		bulk = !Dbulk2.isZero()
		ms2 := Msat2.MSlice()
		defer ms2.Recycle()
		switch {
		case !inter && !bulk:
			cuda.AddExchange(dst2, M2.Buffer(), lex22.Gpu(), ms2, regions.Gpu(), M.Mesh())
		case inter && !bulk:
			//		cuda.AddDMI(dst2, M2.Buffer(), lex22.Gpu(), din22.Gpu(), ms2, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
			cuda.AddDMIAF(dst2, M2.Buffer(), M1.Buffer(), lex22.Gpu(), din22.Gpu(), lex21.Gpu(), din21.Gpu(), lexll.Gpu(), ms2, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		case bulk && !inter:
			cuda.AddDMIBulk(dst2, M2.Buffer(), lex22.Gpu(), dbulk22.Gpu(), ms2, regions.Gpu(), M.Mesh(), OpenBC) // dmi+exchange
		case inter && bulk:
			util.Fatal("Cannot have induced and interfacial DMI at the same time")
		}

		//bex := Bex.MSlice()
		//defer bex.Recycle()
		bex12 := Bex12.MSlice()
		defer bex12.Recycle()
		bex21 := Bex21.MSlice()
		defer bex21.Recycle()
		cuda.AddExchangeAFCell(dst1, dst2, M1.Buffer(), M2.Buffer(), ms1, ms2, bex12, bex21)
		cuda.AddExchangeAFll(dst1, dst2, M1.Buffer(), M2.Buffer(), ms1, ms2, lexll.Gpu(), regions.Gpu(), M.Mesh())
		//cuda.AddExchange(dst1, M2.Buffer(), lexll.Gpu(), ms1, regions.Gpu(), M.Mesh())
		//cuda.AddExchange(dst2, M1.Buffer(), lexll.Gpu(), ms2, regions.Gpu(), M.Mesh())
	*/
}

// Scales the heisenberg exchange interaction between region1 and 2.l
// Scale = 1 means the harmonic mean over the regions of Aex.
func ScaleInterExchangeAex1(region1, region2 int, scale float64) {
	lex21.setScale(region1, region2, scale)
}

// Sets the exchange interaction between region 1 and 2.
func InterExchangeAex1(region1, region2 int, value float64) {
	lex21.setInter(region1, region2, value)
}

// Scales the heisenberg exchange interaction between region1 and 2.
// Scale = 1 means the harmonic mean over the regions of Aex.
func ScaleInterExchangeAex2(region1, region2 int, scale float64) {
	lex22.setScale(region1, region2, scale)
}

// Sets the exchange interaction between region 1 and 2.
func InterExchangeAex2(region1, region2 int, value float64) {
	lex22.setInter(region1, region2, value)
}

// Scales the heisenberg exchange interaction between region1 and 2.
// Scale = 1 means the harmonic mean over the regions of Aex.
func ScaleInterExchangeAexll(region1, region2 int, scale float64) {
	lexll.setScale(region1, region2, scale)
}

// Sets the exchange interaction between region 1 and 2.
func InterExchangeAexll(region1, region2 int, value float64) {
	lexll.setInter(region1, region2, value)
}

// Adds the current exchange field to dst
func AddExchangeField1(dst *data.Slice) {
	size := dst.Size()
	dst2 := cuda.Buffer(3, size)
	defer cuda.Recycle(dst2)
	AddExchangeFieldAF(dst, dst2)
}

// Adds the current exchange field to dst
func AddExchangeField2(dst *data.Slice) {
	size := dst.Size()
	dst2 := cuda.Buffer(3, size)
	defer cuda.Recycle(dst2)
	AddExchangeFieldAF(dst2, dst)
}
