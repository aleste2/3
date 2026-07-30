#include <stdint.h>
#include "float3.h"
#include "amul.h"

// Add uniaxial magnetocrystalline anisotropy field to B.
// http://www.southampton.ac.uk/~fangohr/software/oxs_uniaxial4.html
extern "C" __global__ void
addOSTField(float* __restrict__  Bx, float* __restrict__  By, float* __restrict__  Bz,
                       float* __restrict__  Sx, float* __restrict__  Sy, float* __restrict__  Sz,
                       float* __restrict__ Ms_, float Ms_mul,
                       float* __restrict__ Jex_, float Jex_mul,
                       int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float invMs = inv_Msat(Ms_, Ms_mul, i);
        float  Jex  = amul(Jex_, Jex_mul, i);
        float3 S   = {Sx[i], Sy[i], Sz[i]};
        float3 Ba  = Jex*invMs*S;
        Ba=1e12*Ba;  // Scaling factor due to limitations in max float


        /*if (i==131072) {
          printf("%e %e %e \n",Ba.x,Ba.y,Ba.z);
        }*/

        Bx[i] += Ba.x;
        By[i] += Ba.y;
        Bz[i] += Ba.z;
    }
}
