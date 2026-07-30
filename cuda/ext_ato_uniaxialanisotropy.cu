#include <stdint.h>
#include "float3.h"
#include "amul.h"

// Add uniaxial magnetocrystalline anisotropy field to B.
// http://www.southampton.ac.uk/~fangohr/software/oxs_uniaxial4.html
extern "C" __global__ void
adduniaxialanisotropyato(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                       float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
                       float* __restrict__ Mu_, float Mu_mul,
                       float* __restrict__ Dato_, float Dato_mul,
                       float* __restrict__ ux_, float ux_mul,
                       float* __restrict__ uy_, float uy_mul,
                       float* __restrict__ uz_, float uz_mul,
                       int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 u   = normalized(vmul(ux_, uy_, uz_, ux_mul, uy_mul, uz_mul, i));
        float invMu = inv_Msat(Mu_, Mu_mul, i);
        float  Dato  = amul(Dato_, Dato_mul, i) * invMu;
        float3 m   = {mx[i], my[i], mz[i]};
        float  mu  = dot(m, u);
        float3 Ba  = 2.0f*Dato*(mu)*u;
  //if (i==30000) printf("Before Anis: %f %f %f\n", Bx[i],By[i],Bz[i]);

        Bx[i] += Ba.x;
        By[i] += Ba.y;
        Bz[i] += Ba.z;


  //if (i==30000) printf("Anis: %f %f %f\n", Ba.x,Ba.y,Ba.z);
  //if (i==30000) printf("After Anis: %f %f %f\n", Bx[i],By[i],Bz[i]);
    }
}

