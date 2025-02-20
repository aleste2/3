#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "stencil.h"

extern "C"
__global__ void
calc_Sigmam(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
	float* __restrict__  ux, float* __restrict__  uy, float* __restrict__  uz,
	float* __restrict__  c11_, float c11_mul,
	float* __restrict__  c12_, float c12_mul,
	float* __restrict__ c44_, float c44_mul,
	float* __restrict__ b1_, float b1_mul,
    float* __restrict__ b2_, float b2_mul,
	float wx, float wy, float wz, int Nx, int Ny, int Nz,
    float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
    float* __restrict__  moldx, float* __restrict__  moldy, float* __restrict__  moldz,
    float deltat)

	{

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;
	int I = idx(ix, iy, iz);
	if (ix == 0 || iy == 0|| ix >= Nx || iy >= Ny || iz >= Nz) { // add iz==0&&Nz>1 in the future
		return;
	}

    float C11 = amul(c11_, c11_mul, I);
    float C12 = amul(c12_, c12_mul, I);
    float C44 = amul(c44_, c44_mul, I);
    float B1 = amul(b1_, b1_mul, I);
    float B2 = amul(b2_, b2_mul, I);

    float vx0=ux[I];
    float vy0=uy[I];

	float vx_v=0.0f;
	float vy_v=0.0f;
    //float dmx = (mx[I]-moldx[I])/deltat;
    //float dmy = (my[I]-moldy[I])/deltat;
    float dmx = moldx[I]/deltat;  // Now deltat is 1/GammaLL for direct calculation of dm/dt from torque
    float dmy = moldy[I]/deltat;
    //float dmz = (mz-moldz)/deltat;

    float dxvx,dxvy,dyvx,dyvy;

	int J;
	J=idx(ix-1,iy,iz); // Stress 0 en nx+1 
	vx_v=ux[J];
    vy_v=uy[J];
    dxvx=(vx0-vx_v)*wx;
    dxvy=(vy0-vy_v)*wx;
    
    J=idx(ix,iy-1,iz); // Stress 0 en ny+1
	vx_v=ux[J];
    vy_v=uy[J];
    dyvx=(vx0-vx_v)*wy;
    dyvy=(vy0-vy_v)*wy;

	dstx[I]=C11*dxvx+C12*dyvy-2.0f*B1*mx[I]*dmx;
	dsty[I]=C11*dyvy+C12*dxvx-2.0f*B1*my[I]*dmy;
	dstz[I]=C44*(dyvx+dxvy)-2.0f*B2*(mx[I]*dmy+my[I]*dmx);

    //if (I==257) printf("%e %e \n",dmx,dmy);

}