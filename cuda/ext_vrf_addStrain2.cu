#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "stencil.h"

extern "C"
__global__ void
addStrain2(float* __restrict__  dstx, float* __restrict__  dsty, float* __restrict__  dstz,
    float* __restrict__  sxx, float* __restrict__  syy, float* __restrict__  sxy,
    float* __restrict__  c11_, float c11_mul,
    float* __restrict__  c12_, float c12_mul,
    float* __restrict__ c44_, float c44_mul,
    int N) {

    int I =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (I < N) {

    float factor=1e-9; // To avoid overfloat
    float C11 = amul(c11_, c11_mul, I)*factor;
    float C12 = amul(c12_, c12_mul, I)*factor;
    float C44 = amul(c44_, c44_mul, I)*factor;

    float sigmaxx=sxx[I];
    float sigmayy=syy[I];
    float sigmaxy=sxy[I];

    float C11_2=C11*C11;
    float C11_3=C11*C11*C11;
    float C12_2=C12*C12;
    float C12_3=C12*C12*C12;
    float C44_2=C44*C44;
    float C44_3=C44*C44*C44;

    float f1=factor*(C11_2*C44_3-C12_2*C44_3)/(C11_3*C44_3-3*C11*C12_2*C44_3+2*C12_3*C44_3);
    float f2=factor*(-C11*C12*C44_3+C12_2*C44_3)/(C11_3*C44_3-3*C11*C12_2*C44_3+2*C12_3*C44_3);
    float f3=factor*(C11_3*C44_2-3*C11*C12_2*C44_2+2*C12_3*C44_2)/(C11_3*C44_3-3*C11*C12_2*C44_3+2*C12_3*C44_3);

    dstx[I]=f1*sigmaxx+f2*sigmayy;  // exx
    dsty[I]=f2*sigmaxx+f1*sigmayy;; // eyy
    dstz[I]=0.5*(f3*sigmaxy);  // exy
    }
}