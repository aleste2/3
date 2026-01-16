#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "exchange.h"

extern "C"

 __global__ void
evaldt02Ttausubs(float* __restrict__  templ_,      float* __restrict__ dt0l_,
                float* __restrict__ Tsubsth_, float Tsubsth_mul,
                float* __restrict__ Tausubsth_, float Tausubsth_mul,
                int Nx, int Ny, int Nz,
                float* __restrict__ vol
                ) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    // central cell
    int i = idx(ix, iy, iz);
	float mm = (vol == NULL? 1.0f: vol[i]);

    dt0l_[i]=0.0;

    if (mm!=0)
    {

        float Tsubsth = amul(Tsubsth_, Tsubsth_mul, i);
        float Tausubsth = amul(Tausubsth_, Tausubsth_mul, i);
        float templ = templ_[i];
        if (Tausubsth!=0) {dt0l_[i]=-(templ-Tsubsth)/Tausubsth; }  // Substrate effect on lattice
    }
    else{
        dt0l_[i]=0;
    }

}
