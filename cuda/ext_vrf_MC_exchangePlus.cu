#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"

// See exchange.go for more details.
extern "C" __global__ void
addUexchangePlus(float* __restrict__ Uexplus,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ mxold, float* __restrict__ myold, float* __restrict__ mzold,
            float* __restrict__ Ms_, float Ms_mul,
            float* __restrict__ aLUT2d, uint8_t* __restrict__ regions,
            float wx, float wy, float wz, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix0 = blockIdx.x * blockDim.x + threadIdx.x;
    int iy0 = blockIdx.y * blockDim.y + threadIdx.y;
    int iz0 = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix0 >= Nx || iy0 >= Ny || iz0 >= Nz) {
        return;
    }

    int I0 = idx(ix0, iy0, iz0);
    //Uexplus[I0]=0;

    if (I0==91) printf("Uex=%e ",Uexplus[I0]);


    //Right Neighbour only x derivative
    int ix=ix0+1;
    int iy=iy0;
    int iz=iz0;
    
    if ((ix < Nx) && (iy < Ny) && (iz < Nz) && (ix >=0) && (iy >=0) && (iz >=0) ) {
        // central cell
        int I = idx(ix, iy, iz);
        float3 m0 = make_float3(mxold[I], myold[I], mzold[I]);
        float3 B  = make_float3(0.0,0.0,0.0);

        if (!is0(m0)) {
                uint8_t r0 = regions[I];
                
                int i_;    // neighbor index
                float3 m_; // neighbor mag
                float a__; // inter-cell exchange stiffness

                // left neighbor (current cell, new m)
                i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
                m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
                m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
                a__ = aLUT2d[symidx(r0, regions[i_])];
                B += wx * a__ *(m_ - m0);

                // right neighbor (old m)
                i_  = idx(hclampx(ix+1), iy, iz);
                m_  = make_float3(mxold[i_], myold[i_], mzold[i_]);
                m_  = ( is0(m_)? m0: m_ );
                a__ = aLUT2d[symidx(r0, regions[i_])];
                B += wx * a__ *(m_ - m0);

                float Ms = amul(Ms_, Ms_mul, I);

                Uexplus[I0]-=0.5f*(mxold[I]*B.x+myold[I]*B.y+mzold[I]*B.z);
        }

    }

    //Left Neighbour only x derivative
    ix=ix0-1;
    iy=iy0;
    iz=iz0;
    if ((ix < Nx) && (iy < Ny) && (iz < Nz) && (ix >=0) && (iy >=0) && (iz >=0) ) {
        // central cell
        int I = idx(ix, iy, iz);
        float3 m0 = make_float3(mxold[I], myold[I], mzold[I]);
        float3 B  = make_float3(0.0,0.0,0.0);

        if (!is0(m0)) {
                uint8_t r0 = regions[I];
                //float3 B  = make_float3(0.0,0.0,0.0);

                int i_;    // neighbor index
                float3 m_; // neighbor mag
                float a__; // inter-cell exchange stiffness

                // left neighbor (old m)
                i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
                m_  = make_float3(mxold[i_], myold[i_], mzold[i_]);  // load m
                m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
                a__ = aLUT2d[symidx(r0, regions[i_])];
                B += wx * a__ *(m_ - m0);

                // right neighbor (new m)
                i_  = idx(hclampx(ix+1), iy, iz);
                m_  = make_float3(mx[i_], my[i_], mz[i_]);
                m_  = ( is0(m_)? m0: m_ );
                a__ = aLUT2d[symidx(r0, regions[i_])];
                B += wx * a__ *(m_ - m0);

                float Ms = amul(Ms_, Ms_mul, I);

                Uexplus[I0]-=0.5f*(mxold[I]*B.x+myold[I]*B.y+mzold[I]*B.z);
        }

    }

    /*

    // Back back neighbor only y derivative
    ix=ix0;
    iy=iy0-1;
    iz=iz0;
    if (ix < Nx && iy < Ny && iz < Nz && ix >=0 && iy >=0 && iz >=0 ) {
        // central cell
        int I = idx(ix, iy, iz);
        float3 m0 = make_float3(mxold[I], myold[I], mzold[I]);
        float3 B  = make_float3(0.0,0.0,0.0);

        if (!is0(m0)) {
                uint8_t r0 = regions[I];
                //float3 B  = make_float3(0.0,0.0,0.0);

                int i_;    // neighbor index
                float3 m_; // neighbor mag
                float a__; // inter-cell exchange stiffness

                // back neighbor (old m)
                i_  = idx(ix, lclampy(iy-1), iz);           // clamps or wraps index according to PBC
                m_  = make_float3(mxold[i_], myold[i_], mzold[i_]);  // load m
                m_  = ( is0(m_)? m0: m_ );                  // replace missing non-boundary neighbor
                a__ = aLUT2d[symidx(r0, regions[i_])];
                B += wx * a__ *(m_ - m0);

                // front neighbor new m
                i_  = idx(hclampx(ix+1), iy, iz);
                m_  = make_float3(mx[i_], my[i_], mz[i_]);
                m_  = ( is0(m_)? m0: m_ );
                a__ = aLUT2d[symidx(r0, regions[i_])];
                B += wx * a__ *(m_ - m0);

                float Ms = amul(Ms_, Ms_mul, I);

                Uexplus[I0]+=0.5f*(mxold[I]*B.x+myold[I]*B.y+mzold[I]*B.z);
        }

    }

    */

    // only take vertical derivative for 3D sim
    if (Nz != 1) {

        /*
        // bottom neighbor
        i_  = idx(ix, iy, lclampz(iz-1));
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m0);

        // top neighbor
        i_  = idx(ix, iy, hclampz(iz+1));
        m_  = make_float3(mx[i_], my[i_], mz[i_]);
        m_  = ( is0(m_)? m0: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m0);

        */
    }

    if (I0==91) printf("+Uex+=%e \n",Uexplus[I0]);

}

