package engine

import (
	"fmt"
	"path"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/draw"
	"github.com/mumax/3/httpfs"
	"github.com/mumax/3/util"
)

func init() {
	DeclFunc("ScaledSnapshot", ScaledSnapshot, "Save Scaled image of quantity")
}

func ScaledSnapshot(q Quantity, min,max float64) {
	fname := fmt.Sprintf(OD()+FilenameFormat+"."+SnapshotFormat, NameOf(q), autonum[q])
	s := ValueOf(q)
	defer cuda.Recycle(s)
	data := s.HostCopy() // must be copy (asyncio)
	queOutput(func() { snapshot_sync_scaled(fname, data,min,max) })
	autonum[q]++
}

func snapshot_sync_scaled(fname string, output *data.Slice,min,max float64) {
	f, err := httpfs.Create(fname)
	util.FatalErr(err)
	defer f.Close()
	draw.RenderFormat(f, output, fmt.Sprintf("%f",min), fmt.Sprintf("%f",max), arrowSize, path.Ext(fname))
}

