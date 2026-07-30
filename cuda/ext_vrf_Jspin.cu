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

    float e=1.60218e-19;
    float s, sm, sp,sma,spa;
    int i, im,ip;
    i = idx(ix, iy, iz);
    s = amul(sbar_, sbar_mul, i);
    dstx[i]=0.0f;
    dsty[i]=0.0f;
    dstz[i]=0.0f;

    // x derivatives
    if ((ix > 0) && (ix < Nx - 1)) {
        im = idx(ix-1, iy, iz);
        ip = idx(ix+1, iy, iz);
        sm=amul(sbar_, sbar_mul, im);
        sp=amul(sbar_, sbar_mul, ip);

		if ((s==sm)&&(s==sp)) { 
           dstx[i]+= -(s/e)*(musx[ip] - musx[im]) / (2.0f * wx);
           dsty[i]+= -(s/e)*(musy[ip] - musy[im]) / (2.0f * wx);
           dstz[i]+= -(s/e)*(musz[ip] - musz[im]) / (2.0f * wx);
		} else{
      // Averaged derivative (1/2) on each side (left-right)
		   sma=2.0f*s*sm/(s+sm);
		   spa=2.0f*s*sp/(s+sp);
       //printf("%e \n",musx[im]);
		   dstx[i]+=-(sma/e)*(musx[i] - musx[im]) / (2 * wx)-(spa/e)*(musx[ip] - musx[i]) / (2.0f * wx);
       dsty[i]+=-(sma/e)*(musy[i] - musy[im]) / (2 * wx)-(spa/e)*(musy[ip] - musy[i]) / (2.0f * wx);
       dstz[i]+=-(sma/e)*(musz[i] - musz[im]) / (2 * wx)-(spa/e)*(musz[ip] - musz[i]) / (2.0f * wx);
		}
    }
    if (ix == 0) {
        // Zero en left side (average with 0.5)
        ip = idx(ix+1, iy, iz);
        dstx[i]+= -0.5f*(s/e)*(musx[ip] - musx[i]) / wx;
        dsty[i]+= -0.5f*(s/e)*(musy[ip] - musy[i]) / wx;
        dstz[i]+= -0.5f*(s/e)*(musz[ip] - musz[i]) / wx;
     }
    if (ix == Nx - 1) {
        // Zero en rigth side (average with 0.5)
        //dstx[i]= -(s/e) / wx*(musx[i] - musx[im]);
        im = idx(ix-1, iy, iz);
        dstx[i]+= -0.5f*(s/e)*(musx[i] - musx[im]) / wx ;
        dsty[i]+= -0.5f*(s/e)*(musy[i] - musy[im]) / wx ;
        dstz[i]+= -0.5f*(s/e)*(musz[i] - musz[im]) / wx ;
    }

    // y derivatives
    if ((iy > 0) && (iy < Ny - 1)) {
        im = idx(ix, iy-1, iz);
        ip = idx(ix, iy+1, iz);
        sm=amul(sbar_, sbar_mul, im);
        sp=amul(sbar_, sbar_mul, ip);

		if ((s==sm)&&(s==sp)) { 
           dstx[i]+= -(s/e)*(musx[ip] - musx[im]) / (2.0f * wy);
           dsty[i]+= -(s/e)*(musy[ip] - musy[im]) / (2.0f * wy);
           dstz[i]+= -(s/e)*(musz[ip] - musz[im]) / (2.0f * wy);
		} else{
		   sma=2.0f*s*sm/(s+sm);
		   spa=2.0f*s*sp/(s+sp);
		   dstx[i]+=-(sma/e)*(musx[i] - musx[im]) / (2.0f * wy)-(spa/e)*(musx[ip] - musx[i]) / (2.0f * wy);
       dsty[i]+=-(sma/e)*(musy[i] - musy[im]) / (2.0f * wy)-(spa/e)*(musy[ip] - musy[i]) / (2.0f * wy);
       dstz[i]+=-(sma/e)*(musz[i] - musz[im]) / (2.0f * wy)-(spa/e)*(musz[ip] - musz[i]) / (2.0f * wy);
		}
      }
      if (iy == 0) {
        ip = idx(ix, iy+1, iz);
        dstx[i]+= -0.5f*(s/e)*(musx[ip] - musx[i]) / wy;
        dsty[i]+= -0.5f*(s/e)*(musy[ip] - musy[i]) / wy;
        dstz[i]+= -0.5f*(s/e)*(musz[ip] - musz[i]) / wy;
      }
      if (iy == Ny - 1) {
        im = idx(ix, iy-1, iz);
        dstx[i]+= -0.5f*(s/e)*(musx[i] - musx[im]) / wy;
        dsty[i]+= -0.5f*(s/e)*(musy[i] - musy[im]) / wy;
        dstz[i]+= -0.5f*(s/e)*(musz[i] - musz[im]) / wy;
      }

    // z derivatives
    if (Nz>1) {
     if ((iz > 1) && (iz < Nz - 1)) {
      im = idx(ix, iy, iz-1);
      ip = idx(ix, iy, iz+1);
      sm=amul(sbar_, sbar_mul, im);
      sp=amul(sbar_, sbar_mul, ip);

		if ((s==sm)&&(s==sp)) { 
           dstx[i]+= -(s/e)*(musx[ip] - musx[im]) / (2.0f * wz);
           dsty[i]+= -(s/e)*(musy[ip] - musy[im]) / (2.0f * wz);
           dstz[i]+= -(s/e)*(musz[ip] - musz[im]) / (2.0f * wz);
		} else{
		   sma=2.0f*s*sm/(s+sm);
		   spa=2.0f*s*sp/(s+sp);
		   dstx[i]+=-(sma/e)*(musx[i] - musx[im]) / (2.0f * wz)-(spa/e)*(musx[ip] - musx[i]) / (2.0f * wz);
       dsty[i]+=-(sma/e)*(musy[i] - musy[im]) / (2.0f * wz)-(spa/e)*(musy[ip] - musy[i]) / (2.0f * wz);
       dstz[i]+=-(sma/e)*(musz[i] - musz[im]) / (2.0f * wz)-(spa/e)*(musz[ip] - musz[i]) / (2.0f * wz);
		}
     }
     if (iz == 0) {
        ip = idx(ix, iy, iz+1);
        dstx[i]+= -0.5f*(s/e)*(musx[ip] - musx[i]) / wz;
        dsty[i]+= -0.5f*(s/e)*(musy[ip] - musy[i]) / wz;
        dstz[i]+= -0.5f*(s/e)*(musz[ip] - musz[i]) / wz;
     }
     if (iz == Nz - 1) {
        im = idx(ix, iy, iz-1);
        dstx[i]+= -0.5f*(s/e)*(musx[i] - musx[im]) / wz;
        dsty[i]+= -0.5f*(s/e)*(musy[i] - musy[im]) / wz;
        dstz[i]+= -0.5f*(s/e)*(musz[i] - musz[im]) / wz;
     }
    }
}
