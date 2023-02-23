package engine

// LLB core definitions anf functions

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var (
	TCurie = NewScalarParam("TCurie", "K", "Curie Temperature")
  RenormLLB                   = false
  a1                = NewScalarParam("a1", "a.u.", "Exponent Langevin (T/Tc)^a1") // 0.2-1.4
  a2                = NewScalarParam("a2", "a.u.", "Exponent Langevin (T/Tc)^a2") // 0.2-1.4

)

func unit() {
  	DeclFunc("SetM", SetM, "Adjust m to temperature")
    DeclTVar("Langevin", &Langevin, "Set M(T) to Langevin instead of Brillouin with J=1/2")
    DeclTVar("RenormLLB", &RenormLLB, "Enable/disable remormalize m in LLB")
}

func (m *magnetization) LoadLLBFile(fname string) {
	m.SetLLBArray(LoadFile(fname))
}

func (b *magnetization) SetLLBArray(src *data.Slice) {
	if src.Size() != b.Mesh().Size() {
		src = data.Resample(src, b.Mesh().Size())
	}
	data.Copy(b.Buffer(), src)
}

func SetM() {
	TCurie := TCurie.MSlice()
	defer TCurie.Recycle()
	if solvertype == 26 {
		Temp := Temp.MSlice()
		defer Temp.Recycle()
		cuda.InitmLLB(M.Buffer(), Temp, TCurie, Langevin)
	}
	if solvertype == 27 {
		cuda.InitmLLBJH(M.Buffer(), TempJH.temp, TCurie, Langevin)
	}
	if solvertype == 28 {
		cuda.InitmLLBJH(M.Buffer(), Ts.temp, TCurie, Langevin)
	}
	if solvertype == 29 {
		cuda.InitmLLBJH(M.Buffer(), Te.temp, TCurie, Langevin)
	}
}
