package engine

// Simple averager for vector field

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Epecemos usando tipos definidos y ya veremos luego. Inicialmente 2D
var (
	m_avg = NewVectorField("m_avg", "a.u.", "Averaged m", Averagem)
	navg  = 1
	mask  *data.Slice // buffer for slow Te evolucion
)

func init() {
	DeclTVar("Navg", &navg, "Number of neighbours for averages")
}

// Averager
func Averagem(dst *data.Slice) {
	Averager(dst, M.Buffer())
}

func Averager(dst, src *data.Slice) {
	if mask == nil {
		size := M.Mesh().Size()
		mask = cuda.NewSlice(1, size)
		//defer cuda.Recycle(mask)
		cuda.AddDotProduct(mask, 1.0, M.Buffer(), M.Buffer())
	}
	cuda.Averager(dst, src, mask,
		navg)
}
