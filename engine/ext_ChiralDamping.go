package engine

// Chiral damping Core

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Chiral damping variables and functions
var (
	AlphaChiral      = NewScalarParam("alphaChiral", "a.u.", "Chiral damping constant")
	AlphaChiralField = NewScalarField("alphaChiralField", "a.u.", "Alpha Chiral Local values", calculateAlphaChiral)
)

func calculateAlphaChiral(dst *data.Slice) {
	alphaC := AlphaChiral.MSlice()
	defer alphaC.Recycle()
	ku1 := Ku1.MSlice()
	defer ku1.Recycle()
	cuda.AlphaChiralEvaluate(dst, M.Buffer(), alphaC, ku1, lex2.Gpu(), regions.Gpu(), M.Mesh())
}
