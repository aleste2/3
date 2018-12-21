package engine

// import ("math")

func init() {
	DeclFunc("ext_mMaxPos", mMaxPos, "Skyrmion core position (x,y) + radio (z)")
}

func mMaxPos(comp int) []float64 {

	m := M.Buffer()
	m_z := m.Comp(comp).HostCopy().Scalars()
	s := m.Size()
	Nx, Ny, Nz := s[X], s[Y], s[Z]


        acumM := float64(0)
        indexx :=int(0)
        indexy :=int(0)
        indexz :=int(0)

	for z := 0; z < Nz; z++ {
		// Avoid the boundaries so the neighbor interpolation can't go out of bounds.
		for y := 1; y < Ny-1; y++ {
			for x := 1; x < Nx-1; x++ {
				m := m_z[z][y][x]
				if (float64(abs(m)) > acumM) {
					acumM=float64(abs(m))
					indexx=x
                                        indexy=y
                                        indexz=z
				}

			}
		}
	}

	pos := make([]float64, 3)

	c := Mesh().CellSize()
	pos[X] = c[X]*float64(indexx)
	pos[Y] = c[Y]*float64(indexy)
	pos[Z] = c[Z]*float64(indexz)

	pos[X] += GetShiftPos() // add simulation window shift
        pos[Y] += GetShiftPosY() // add simulation window shift
	return pos
}






