//#include <stdint.h>
#include "amul.h"
#include "float3.h"
#include "stencil.h"
#include "constants.h"

extern "C"
__global__ void
addJspin(
    float* __restrict__ dstx, float* __restrict__ dsty, float* __restrict__ dstz,
    float* __restrict__ musx, float* __restrict__ musy, float* __restrict__ musz,
    float* __restrict__ sbar_, float sbar_mul,
    float wx, float wy, float wz, int Nx, int Ny, int Nz,
    float* __restrict__ vol
) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
        return;
    }

    float e=1.6e-19;
    float s, sm, sp,sma,spa;
    int i, im,ip;
    i = idx(ix, iy, iz);
    s = amul(sbar_, sbar_mul, i);
    dstx[i]=0;
    dsty[i]=0;
    dstz[i]=0;

    // x derivatives
    if ((ix > 1) && (ix < Nx - 1)) {
        im = idx(ix-1, iy, iz);
        ip = idx(ix+1, iy, iz);
        sm=amul(sbar_, sbar_mul, im);
        sp=amul(sbar_, sbar_mul, ip);

		if ((s==sm)&&(s==sp)) { 
           dstx[i]+= -(s/e)*(musx[ip] - musx[im]) / (2 * wx);
           dsty[i]+= -(s/e)*(musy[ip] - musy[im]) / (2 * wx);
           dstz[i]+= -(s/e)*(musz[ip] - musz[im]) / (2 * wx);
		} else{
		   sma=2.0f*s*sm/(s+sm);
		   spa=2.0f*s*sp/(s+sp);
		   dstx[i]+=-(sma/e)*(musx[i] - musx[ip]) / (2 * wx)-(spa/e)*(musx[ip] - musx[i]) / (2 * wx);
           dsty[i]+=-(sma/e)*(musy[i] - musy[ip]) / (2 * wx)-(spa/e)*(musy[ip] - musy[i]) / (2 * wx);
           dstz[i]+=-(sma/e)*(musz[i] - musz[ip]) / (2 * wx)-(spa/e)*(musz[ip] - musz[i]) / (2 * wx);
		}
      }
      if (ix == 0) {
        dstx[i]+= -(s/e)*(musx[ip] - musx[i]) / wx;
        dsty[i]+= -(s/e)*(musy[ip] - musy[i]) / wx;
        dstz[i]+= -(s/e)*(musz[ip] - musz[i]) / wx;
      }
      if (ix == Nx - 1) {
        dstx[i]+= -(s/e)*(musx[i] - musx[im]) / wx;
        dsty[i]+= -(s/e)*(musy[i] - musy[im]) / wx;
        dstz[i]+= -(s/e)*(musz[i] - musz[im]) / wx;
      }
/*
    // y derivatives
    if ((iy > 1) && (iy < Ny - 1)) {
        im = idx(ix, iy-1, iz);
        ip = idx(ix, iy+1, iz);
        sm=amul(sbar_, sbar_mul, im);
        sp=amul(sbar_, sbar_mul, ip);

		if ((s==sm)&&(s==sp)) { 
           dstx[i]+= -(s/e)*(musx[ip] - musx[im]) / (2 * wy);
           dsty[i]+= -(s/e)*(musy[ip] - musy[im]) / (2 * wy);
           dstz[i]+= -(s/e)*(musz[ip] - musz[im]) / (2 * wy);
		} else{
		   sma=2.0f*s*sm/(s+sm);
		   spa=2.0f*s*sp/(s+sp);
		   dstx[i]+=-(sma/e)*(musx[i] - musx[ip]) / (2 * wy)-(spa/e)*(musx[ip] - musx[i]) / (2 * wy);
           dsty[i]+=-(sma/e)*(musy[i] - musy[ip]) / (2 * wy)-(spa/e)*(musy[ip] - musy[i]) / (2 * wy);
           dstz[i]+=-(sma/e)*(musz[i] - musz[ip]) / (2 * wy)-(spa/e)*(musz[ip] - musz[i]) / (2 * wy);
		}
      }
      if (iy == 0) {
        dstx[i]+= -(s/e)*(musx[ip] - musx[i]) / wy;
        dsty[i]+= -(s/e)*(musy[ip] - musy[i]) / wy;
        dstz[i]+= -(s/e)*(musz[ip] - musz[i]) / wy;
      }
      if (iy == Ny - 1) {
        dstx[i]+= -(s/e)*(musx[i] - musx[im]) / wy;
        dsty[i]+= -(s/e)*(musy[i] - musy[im]) / wy;
        dstz[i]+= -(s/e)*(musz[i] - musz[im]) / wy;
      }
*/

/*
    // z derivatives
    if (Nz>1) {
     if ((iz > 1) && (iz < Nz - 1)) {
      im = idx(ix, iy, iz-1);
      ip = idx(ix, iy, iz+1);
      sm=amul(sbar_, sbar_mul, im);
      sp=amul(sbar_, sbar_mul, ip);

		if ((s==sm)&&(s==sp)) { 
           dstx[i]+= -(s/e)*(musx[ip] - musx[im]) / (2 * wz);
           dsty[i]+= -(s/e)*(musy[ip] - musy[im]) / (2 * wz);
           dstz[i]+= -(s/e)*(musz[ip] - musz[im]) / (2 * wz);
		} else{
		   sma=2.0f*s*sm/(s+sm);
		   spa=2.0f*s*sp/(s+sp);
		   dstx[i]+=-(sma/e)*(musx[i] - musx[ip]) / (2 * wz)-(spa/e)*(musx[ip] - musx[i]) / (2 * wz);
           dsty[i]+=-(sma/e)*(musy[i] - musy[ip]) / (2 * wz)-(spa/e)*(musy[ip] - musy[i]) / (2 * wz);
           dstz[i]+=-(sma/e)*(musz[i] - musz[ip]) / (2 * wz)-(spa/e)*(musz[ip] - musz[i]) / (2 * wz);
		}
     }
     if (iz == 0) {
        dstx[i]+= -(s/e)*(musx[ip] - musx[i]) / wz;
        dsty[i]+= -(s/e)*(musy[ip] - musy[i]) / wz;
        dstz[i]+= -(s/e)*(musz[ip] - musz[i]) / wz;
     }
     if (iz == Nz - 1) {
        dstx[i]+= -(s/e)*(musx[i] - musx[im]) / wz;
        dsty[i]+= -(s/e)*(musy[i] - musy[im]) / wz;
        dstz[i]+= -(s/e)*(musz[i] - musz[im]) / wz;
     }
    }
     */
}
