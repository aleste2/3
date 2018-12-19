#include "amul.h"
#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"

// normalize vector {vx, vy, vz} to unit length, unless length or vol are zero.
extern "C" __global__ void
addexchangeAfll(float* __restrict__ dst1x, float* __restrict__ dst1y, float* __restrict__ dst1z,
	float* __restrict__ dst2x, float* __restrict__ dst2y, float* __restrict__ dst2z,
	float* __restrict__ m1x, float* __restrict__ m1y, float* __restrict__ m1z,
	float* __restrict__ m2x, float* __restrict__ m2y, float* __restrict__ m2z,
	float* __restrict__  Ms1_, float  Ms1_mul,
	float* __restrict__  Ms2_, float  Ms2_mul,
        float* __restrict__ aLUT2d, uint16_t* __restrict__ regions,
        float wx, float wy, float wz, int Nx, int Ny, int Nz, uint16_t PBC) {


    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }


    // central cell
    int I = idx(ix, iy, iz);
    float3 m1 = make_float3(m1x[I], m1y[I], m1z[I]);
    float3 m2 = make_float3(m2x[I], m2y[I], m2z[I]);
    float invMs1 = inv_Msat(Ms1_, Ms1_mul, I);
    float invMs2 = inv_Msat(Ms2_, Ms2_mul, I);
 
    // First lattice

    if (!is0(m2)) {

    uint16_t r0 = regions[I];
    float3 B  = make_float3(0.0, 0.0, 0.0);

    int i_;    // neighbor index
    float3 m_; // neighbor mag
    float a__; // inter-cell exchange stiffness

    // left neighbor
    i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
    m_  = make_float3(m2x[i_], m2y[i_], m2z[i_]);  // load m
    m_  = ( is0(m_)? m2: m_ );                  // replace missing non-boundary neighbor
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ *(m_ - m2);

    // right neighbor
    i_  = idx(hclampx(ix+1), iy, iz);
    m_  = make_float3(m2x[i_], m2y[i_], m2z[i_]);
    m_  = ( is0(m_)? m2: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wx * a__ *(m_ - m2);

    // back neighbor
    i_  = idx(ix, lclampy(iy-1), iz);
    m_  = make_float3(m2x[i_], m2y[i_], m2z[i_]);
    m_  = ( is0(m_)? m2: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m2);

    // front neighbor
    i_  = idx(ix, hclampy(iy+1), iz);
    m_  = make_float3(m2x[i_], m2y[i_], m2z[i_]);
    m_  = ( is0(m_)? m2: m_ );
    a__ = aLUT2d[symidx(r0, regions[i_])];
    B += wy * a__ *(m_ - m2);

    // only take vertical derivative for 3D sim
    if (Nz != 1) {
        // bottom neighbor
        i_  = idx(ix, iy, lclampz(iz-1));
        m_  = make_float3(m2x[i_], m2y[i_], m2z[i_]);
        m_  = ( is0(m_)? m2: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m2);

        // top neighbor
        i_  = idx(ix, iy, hclampz(iz+1));
        m_  = make_float3(m2x[i_], m2y[i_], m2z[i_]);
        m_  = ( is0(m_)? m2: m_ );
        a__ = aLUT2d[symidx(r0, regions[i_])];
        B += wz * a__ *(m_ - m2);
    }

    dst1x[I] += B.x*invMs1;
    dst1y[I] += B.y*invMs1;
    dst1z[I] += B.z*invMs1;
    }
}
