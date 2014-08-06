#include "MathUtils.cuh"
#include <cstdio>


__device__ float force()
{

}

__global__ void cutest(float3* pos, float3* vel, float* mass)
{
	int x = threadIdx.x;
/*
	printf("pos: %f, %f, %f\n\n", pos[x].x,  pos[x].y,  pos[x].z);
	printf("vel: %f, %f, %f\n\n", vel[x].x,  vel[x].y,  vel[x].z);
	printf("mass: %f\n\n", mass[x]);
*/
	//verts[x].x -= 0.0001f;

}

void runPhysics(float3* devPos, float3* devVel, float* devMass)
{

	cutest<<<1, 3>>>(devPos, devVel, devMass);
}