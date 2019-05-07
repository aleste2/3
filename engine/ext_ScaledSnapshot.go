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
	DeclFunc("AutoSnapshotScaled", AutoScaledSnapshot, "Auto save image of quantity every period (s).")

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

// For Autosave

// Register quant to be auto-saved as image, every period.
func AutoScaledSnapshot(q Quantity, min,max ,period float64) {
	autoSaveScaled(q, min,max,period, ScaledSnapshot)
}

// register save(q) to be called every period
func autoSaveScaled(q Quantity, min, max, period float64, save func(Quantity,float64,float64)) {
	if period == 0 {
		delete(output, q)
	} else {
		outputScaled[q] = &autosavescaled{period, Time, -1, min,max,save} // init count to -1 allows save at t=0
	}
}

// keeps info needed to decide when a quantity needs to be periodically saved
type autosavescaled struct {
	period float64        // How often to save
	start  float64        // Starting point
	count  int            // Number of times it has been autosaved
	min float64
	max float64
	save   func(Quantity,float64,float64) // called to do the actual save
}

func (a *autosavescaled) needSave() bool {
	t := Time - a.start
	return a.period != 0 && t-float64(a.count)*a.period >= a.period
}

