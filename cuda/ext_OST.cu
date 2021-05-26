#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"

extern "C"


__global__ void
evaldtOST(	float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
  float* __restrict__ Sx, float* __restrict__ Sy, float* __restrict__ Sz,
  float* __restrict__ dy1x, float* __restrict__ dy1y, float* __restrict__ dy1z,
                float* __restrict__ tau_, float tau_mul,
                float* __restrict__ Jex_, float Jex_mul,
                float* __restrict__ R_, float R_mul,
                float* __restrict__ dirx_, float dirx_mul,
                float* __restrict__ diry_, float diry_mul,
                float* __restrict__ dirz_, float dirz_mul,
            		float wx, float wy, float wz, int Nx, int Ny, int Nz
                ) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }
    // central cell
    int i = idx(ix, iy, iz);
    float3 m0 = make_float3(mx[i], my[i], mz[i]);
    float3 S0 = make_float3(Sx[i], Sy[i], Sz[i]);

    float mm=dot(m0,m0);
    if (mm!=0)
    {
    float3 dir = vmul(dirx_, diry_, dirz_, dirx_mul, diry_mul, dirz_mul, i);
    float tau = amul(tau_, tau_mul, i);
    float Jex = amul(Jex_, Jex_mul, i);
    float R = amul(R_, R_mul, i);

    float3 SxM=cross(S0,m0);
    float3 torque=-Jex/1.054e-34*SxM+R*dir-1.0f/tau*S0;

//    if (i==131000) printf("%e %e %e %e %e %e %e\n",torque.x,torque.y,torque.z,S0.x,S0.y,S0.z,R);

    dy1x[i]=torque.x;
    dy1y[i]=torque.y;
    dy1z[i]=torque.z;
    }
}
