#include "MathUtils.cuh"
#include <cstdio>

#define G 0.0000000000667f //Nm^2/kg^2
#define EP 0.05f

/*
__device__ float force()
{

}
*/



__global__ void cuNBody(float3* pos, float3* vel, float* mass, int numPoints)
{

	int x = threadIdx.x;

	//pos[x].x += G * 100000;

//	float3 a_i = make_float3(0, 0, 0); 



}

__global__ void allPairsAcceleration(float3* pos, float3* acc, float* mass, int numPoints)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < numPoints)
	{
		float3 a_i = make_float3(0, 0, 0); //Net acceleration on object i

		float3 posi = pos[i];

		for(int j = 0; j < numPoints; j++)
		{
			if(j != i)
			{
				float3 r_ij = pos[j] - posi;
				float magr = magnitude(r_ij);

				a_i = a_i + __powf(magr*magr + EP*EP, -3.0f/2.0f) * r_ij * mass[j];

				//
			}
		}

		acc[i] = a_i * G;
	}
}

__global__ void integrate(float3* pos, float3* vel, float3* acc, int numPoints)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < numPoints)
	{
		vel[i] = vel[i] + acc[i] * 0.01f;
		pos[i] = pos[i] + vel[i] * 0.01f;
	}
}

void runPhysics(float3* devPos, float3* devVel, float3* devAcc, float* devMass, int numPoints)
{
	dim3 blockSize = dim3(512);
	dim3 gridSize = dim3((numPoints+blockSize.x-1)/blockSize.x);
	allPairsAcceleration<<<gridSize, blockSize>>>(devPos, devAcc, devMass, numPoints);

	integrate<<<gridSize, blockSize>>>(devPos, devVel, devAcc, numPoints);
	//cuNBody<<<1, 3>>>(devPos, devVel, devMass, numPoints);
}