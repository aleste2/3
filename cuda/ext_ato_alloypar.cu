#include <stdint.h>
#include "exchange.h"
#include "float3.h"
#include "stencil.h"
#include "amul.h"
#include "math.h"

// See exchange.go for more details.
extern "C" __global__ void
alloyparcuda(uint8_t host,uint8_t alloy, float percent, float* random, uint8_t* regions,int N) {
    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {
	//printf("%i %i %f %i\n",host,alloy,random[i],regions[i]);
	if ((regions[i]==host)&&(random[i]<=percent)) {regions[i]=alloy;}

    }
}

