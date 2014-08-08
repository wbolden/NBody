#include "MathUtils.cuh"
#include <cstdio>

/*
__device__ float force()
{

}
*/

__global__ void cutest(float3* pos, float3* vel, float* mass)
{
	int x = threadIdx.x;

	//pos[x].x -= 0.0001f;

	vel[x].x += 0.0001f;
}

__global__ void cuNBody(float3* pos, float3* vel, float* mass)
{

}

void runPhysics(float3* devPos, float3* devVel, float* devMass)
{
	cuNBody<<<1, 3>>>(devPos, devVel, devMass);
}