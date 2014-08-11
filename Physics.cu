#include "MathUtils.cuh"
#include <cstdio>

#define G 0.0000000000667f //Nm^2/kg^2
#define EP 0.005f
#define TIMESTEP 0.01f

__global__ void allPairsAcceleration(float3* pos, float3* acc, float* mass, int numPoints)
{
	extern __shared__ float4 bodyInfo[];

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < numPoints)
	{
		float3 a_i = make_float3(0, 0, 0); //Net acceleration on object i
		float3 posi = pos[i];

		for(int tile = 0; tile < gridDim.x; tile++)
		{
			int localIndex = tile * blockDim.x + threadIdx.x;

			bodyInfo[threadIdx.x] = make_float4(pos[localIndex].x, pos[localIndex].y, pos[localIndex].z, mass[localIndex]);
			__syncthreads();

			for(int j = 0; j < blockDim.x; j++)
			{
				if(i != tile * blockDim.x + j)
				{
					float3 r_ij = bodyInfo[j] - posi;
					float magr = fmagnitude(r_ij);

					a_i = a_i + __powf(magr*magr + EP*EP, -3.0f/2.0f) * r_ij * bodyInfo[j].w;
				}
			}
			__syncthreads();
		}
		acc[i] = a_i * G;

	}



	/*
	if(i < numPoints)
	{
		float3 a_i = make_float3(0, 0, 0); //Net acceleration on object i

		float3 posi = pos[i];

		for(int j = 0; j < numPoints; j++)
		{
			if(j != i)
			{
				float3 r_ij = pos[j] - posi;
				float magr = fmagnitude(r_ij);

				a_i = a_i + __powf(magr*magr + EP*EP, -3.0f/2.0f) * r_ij * mass[j];
			}
		}

		acc[i] = a_i * G;
	}
	*/
	
}

__global__ void integrateEuler(float3* pos, float3* vel, float3* acc, int numPoints)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < numPoints)
	{
		vel[i] = vel[i] + acc[i] * TIMESTEP;
		pos[i] = pos[i] + vel[i] * TIMESTEP;
	}
}

void runPhysics(float3* devPos, float3* devVel, float3* devAcc, float* devMass, int numPoints)
{
	dim3 blockSize = dim3(512);
	dim3 gridSize = dim3((numPoints+blockSize.x-1)/blockSize.x);
	int smem = sizeof(float4)*blockSize.x;

	allPairsAcceleration<<<gridSize, blockSize, smem>>>(devPos, devAcc, devMass, numPoints);

	integrateEuler<<<gridSize, blockSize>>>(devPos, devVel, devAcc, numPoints);

}