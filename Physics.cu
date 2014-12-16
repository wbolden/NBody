#include "MathUtils.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cstdio>

#define G 6.67e-11f //Nm^2/kg^2
#define EP 0.01f
#define EPS EP*EP
#define TIMESTEP 0.05f

#define LEAF -1
#define NODE 1
#define UNUSED 0
#define LOCKED -2

__device__ float theta = 1000.2f;

__device__ __host__ struct bounds
{
	float3 min;
	float3 max;
};

__device__ struct bhInfo
{
	float3 centerOfMass;
	float mass;
	int intNumBodies;
};

float3* minPos = 0;
float3* maxPos = 0;

dim3 blockSize = dim3(512);
dim3 gridSize;

bool init = false;

//Octree
#define numArr 1000000

int* devNodeTypes;
int* devIndices;
bounds* devNodeBounds;
int* devSupernode;

bhInfo* devInfo;

//int* defaultNodeTypes;
//int* defaultIndices;
//bounds* defaultBounds;




__global__ void allPairsNormal(float3* pos, float3* acc, float* mass, int numPoints, float3* vel)
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

			#pragma unroll 32

			for(int j = 0; j < blockDim.x; j++)
			{
				if(i != tile * blockDim.x + j)
				{	
					float3 r_ij = bodyInfo[j] - posi;

					float dotr = dot(r_ij) + EPS;

					 a_i = a_i + rsqrtf(dotr*dotr*dotr ) * bodyInfo[j].w * r_ij;

				//	a_i = a_i + rsqrtf(dotr*dotr*dotr) * bodyInfo[j].w * r_ij;

				//	a_i = a_i + atanf(sqrtf(dotr)) * bodyInfo[j].w * r_ij*rsqrtf(dotr);

					// a_i.y = 0;

					
				}
			}
			__syncthreads();
		}
		acc[i] = a_i * G;

		vel[i] = vel[i] + acc[i] * TIMESTEP;
		pos[i] = pos[i] + vel[i] * TIMESTEP;
	}
}

__global__ void integrateEuler(float3* pos, float3* vel, float3* acc, int numPoints)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if(i < numPoints)
	{
		acc[i] = acc[i] * G;
		vel[i] = vel[i] + acc[i] * TIMESTEP;
		pos[i] = pos[i] + vel[i] * TIMESTEP;
	}
}

__global__ void cuGet3DMinMax(float3* pos, int numInputs, float3* minPos, float3* maxPos, bool first)
{
	extern __shared__ float3 sdata[];
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tidMax = threadIdx.x;
	int tidMin = blockDim.x + threadIdx.x;



	if(i < numInputs)
	{
		if(first)
		{
			sdata[tidMax] = pos[i];
			sdata[tidMin] = sdata[tidMax];
		}
		else
		{
			sdata[tidMax] = maxPos[i];
			sdata[tidMin] = minPos[i];
		}
	}
	else
	{
		sdata[tidMax] = make_float3(-9999999, -9999999, -9999999);
		sdata[tidMin] = make_float3(9999999, 9999999, 9999999);
	}

	__syncthreads();

	for(int s = blockDim.x/2; s > 0; s/=2)
	{
		if(threadIdx.x < s && i < numInputs)
		{
			sdata[tidMax].x = max(sdata[tidMax].x, sdata[tidMax + s].x);
			sdata[tidMax].y = max(sdata[tidMax].y, sdata[tidMax + s].y);
			sdata[tidMax].z = max(sdata[tidMax].z, sdata[tidMax + s].z);

			sdata[tidMin].x = min(sdata[tidMin].x, sdata[tidMin + s].x);
			sdata[tidMin].y = min(sdata[tidMin].y, sdata[tidMin + s].y);
			sdata[tidMin].z = min(sdata[tidMin].z, sdata[tidMin + s].z);
		}
		__syncthreads();
	}


	if(threadIdx.x == 0)
	{
		maxPos[blockIdx.x] = sdata[tidMax];
		minPos[blockIdx.x] = sdata[tidMin];
	}

}

__global__ void printf3(float3* pos)
{
	printf("%f  %f  %f\n", pos[0].x, pos[0].y, pos[0].z);
}

__device__ int lastins = 1;
__device__ int leafAlloc(int numLeaves, int* nodeArray, bounds* nodeBounds, int currentPos, int* supernode)
{
	int i = lastins;

	bool lockAcquired = false;
	while(!lockAcquired)
	{
		if(nodeArray[i] == UNUSED)
		{
				if (UNUSED == atomicCAS(&nodeArray[i], UNUSED, LEAF)) //Attempt to lock position i as a leaf
				{
					lockAcquired = true;
				}
				else
				{
					i+=8; 
				}
		}
		else
		{
			i += 8;
		}
	}

	lastins = i;

	for(int j = 0; j < 8; j++)
	{
		nodeArray[i+j] = LEAF;
		supernode[i+j] = currentPos;

		//z |4
		if((j & 4) == 4)
		{
			//upper half
			nodeBounds[i+j].min.z = (nodeBounds[currentPos].max.z + nodeBounds[currentPos].min.z)/2;
			nodeBounds[i+j].max.z = nodeBounds[currentPos].max.z;
		}
		else
		{
				//lower half
			nodeBounds[i+j].min.z = nodeBounds[currentPos].min.z;
			nodeBounds[i+j].max.z = (nodeBounds[currentPos].max.z + nodeBounds[currentPos].min.z)/2;
		}

		//y |2
		if((j & 2) == 2)
		{
			//upper half
			nodeBounds[i+j].min.y = (nodeBounds[currentPos].max.y + nodeBounds[currentPos].min.y)/2;
			nodeBounds[i+j].max.y = nodeBounds[currentPos].max.y;
		}
		else
		{
			//lower half
			nodeBounds[i+j].min.y = nodeBounds[currentPos].min.y;
			nodeBounds[i+j].max.y = (nodeBounds[currentPos].max.y + nodeBounds[currentPos].min.y)/2;
		}

		//x |1
		if((j & 1) == 1)
		{
			//upper half
			nodeBounds[i+j].min.x = (nodeBounds[currentPos].max.x + nodeBounds[currentPos].min.x)/2;
			nodeBounds[i+j].max.x = nodeBounds[currentPos].max.x;
		}
		else
		{
			//lower half
			nodeBounds[i+j].min.x = nodeBounds[currentPos].min.x;
			nodeBounds[i+j].max.x = (nodeBounds[currentPos].max.x + nodeBounds[currentPos].min.x)/2;
		}
	}
	
	return i;
}

__device__  int getNodeIndex(bounds bounds, float3 point)
{
	int index = 0;

	if((bounds.min.x + bounds.max.x)/2 < point.x) index |= 1;
	if((bounds.min.y + bounds.max.y)/2 < point.y) index |= 2;
	if((bounds.min.z + bounds.max.z)/2 < point.z) index |= 4;

	return index;
}

__device__  void branch(int* nodeArray, int* indices, bounds* nodeBounds, int currentPos, int* supernode)
{
	indices[currentPos] = leafAlloc(8, nodeArray, nodeBounds, currentPos, supernode);
}

__global__ void insert(int* nodeArray, int* indices, bounds* nodeBounds, float3* data, int numItems, int* supernode)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int depth = 0;
	int tries = 1;

	if(i < numItems)
	{
		bool inserted = false;
		int currentNode = 0;
		int nodeIndex;
	
		while(!inserted)
		{
			
			nodeIndex = getNodeIndex(nodeBounds[currentNode], data[i]); //The offset from the index of the current node
			int subnodeIndex = indices[currentNode] + nodeIndex; //The node the data will attempt to be inserted into

			if(nodeArray[subnodeIndex] == NODE)
			{
				//Follow the node
				currentNode = subnodeIndex;
				depth++;
			}
			else if(LEAF == atomicCAS(&nodeArray[subnodeIndex], LEAF, LOCKED)) // Lock obtained
			{
			//	numWrites[i] +=1;

				if(indices[subnodeIndex] == UNUSED)
				{
					indices[subnodeIndex] = i;
					inserted = true;
					nodeArray[subnodeIndex] = LEAF; //release lock
				}
				else
				{				
					int oldIndex = indices[subnodeIndex]; //save the old index
					branch(nodeArray, indices, nodeBounds, subnodeIndex, supernode);

					//insert oldIndex into one of the new leaves
					int leafIndex = indices[subnodeIndex]+getNodeIndex(nodeBounds[subnodeIndex], data[oldIndex]);
					indices[leafIndex] = oldIndex;
					__threadfence();
					nodeArray[subnodeIndex] = NODE; //releases the lock on the cell

					//set the current node to the subnodeIndex
					currentNode = subnodeIndex;
				}
			}
			else
			{
				tries++;
				//__syncthreads();
			}
			__syncthreads();
			
		}
	}
//	numWrites[i] = tries;
}

__global__ void beginTree(int* nodeType, int* indices, bounds* nodeBounds, int* supernode, float3* minPos, float3* maxPos)
{
	nodeBounds[0].max = maxPos[0];
	nodeBounds[0].min = minPos[0];
	nodeType[0] = NODE;

	branch(nodeType, indices, nodeBounds, 0, supernode);
}

__device__ int getPointIndex(int* nodeArray, int* indices, bounds* nodeBounds, float3 point)
{
	int currentNode = 0;
	int i = 0;

	while(nodeArray[currentNode] == NODE)
	{
		i++;
		currentNode = indices[currentNode] + getNodeIndex(nodeBounds[currentNode], point);
	}
	return currentNode;
}

__global__ void sumMassNaive(int* nodeArray, int* indices, bounds* nodebounds, int* supernode, bhInfo* info, float3* data, float* mass, int numItems)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i < numItems)
	{
		int lastNode = getPointIndex(nodeArray, indices, nodebounds, data[i]);
		float bodyMass = mass[i];

		info[lastNode].mass = bodyMass;
		info[lastNode].centerOfMass = data[i];

	//	info[lastNode].centerOfMass.y -= 0.01f;

	//	info[lastNode].centerOfMass.x *= bodyMass;
	//	info[lastNode].centerOfMass.y *= bodyMass;
	//	info[lastNode].centerOfMass.z *= bodyMass;
		
		info[lastNode].intNumBodies = 1;
		while(supernode[lastNode] != UNUSED)
		{
			int index = supernode[lastNode];
			
			atomicAdd(&info[index].mass, bodyMass);
			atomicAdd(&info[index].centerOfMass.x, data[i].x*bodyMass);
			atomicAdd(&info[index].centerOfMass.y, data[i].y*bodyMass);
			atomicAdd(&info[index].centerOfMass.z, data[i].z*bodyMass);
			lastNode = supernode[lastNode];
		}
	}
}

__global__ void divideMassNaive(bhInfo* info, int numItems)
{

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(i < numItems)
	{
		if(info[i].mass != 0 && info[i].intNumBodies == 0)
		{
			info[i].centerOfMass.x /= info[i].mass;
			info[i].centerOfMass.y /= info[i].mass;
			info[i].centerOfMass.z /= info[i].mass;
		}
	}
}

__global__ void cuBarnesHut(int* nodeArray, int* indices, bounds* nodebounds, int* supernode, bhInfo* info, float3* pos, float3* acc, float* mass, int numItems)
{
	int ind = blockIdx.x * blockDim.x + threadIdx.x;

	int iVal[300];
	int cPos[300];
	int sp = 0;

	if(ind < numItems)
	{
		int currentPos = 1;
		for(int i = 0; i < 8; i++)
		{
			float3 r_ij = info[currentPos+i].centerOfMass - pos[ind];

			if(info[currentPos+i].mass > 0 && !(r_ij.x == 0 && r_ij.y == 0 && r_ij.z == 0) )
			{
				float ratio = -1;

				float dotr = dot(r_ij);
				bool same = (dotr == 0);

				int tst = 0;

				if(dotr == 0)
				{
					
					tst++;
				}

				if(!same)
				{		
					ratio = (nodebounds[currentPos+i].max.x - nodebounds[currentPos+i].min.x)*rsqrtf(dotr);
				}

				if((ratio < theta || nodeArray[currentPos+i] == LEAF) && !same)
				{
					dotr += EPS;
					float cmass = info[currentPos+i].mass;
					acc[ind] = acc[ind] + rsqrtf(dotr*dotr*dotr ) * cmass * r_ij;
					//done
				}
				else
				{
					iVal[sp] = i;
					cPos[sp] = currentPos;
					sp++;

					currentPos = indices[currentPos+i];
					i = 0;
				}
			}

			if(i == 7 && sp > 0)
			{
				do
				{
					sp--;
					i = iVal[sp];
					currentPos = cPos[sp];
				}
				while(sp > 0 && i == 7); //keep popping until you reach a non-7 value or the stack runs out
			}
		}

	}
}

void get3DMinMax(float3* devPoints, int numPoints, float3* min, float3* max)
{
	dim3 gridSize;
	bool first = true;

	int smem = sizeof(float3)*blockSize.x*2; 
	int numInputs = numPoints; //The first iteration looks at every position
	
	do
	{
		gridSize = dim3((numInputs+blockSize.x-1)/blockSize.x);

		cuGet3DMinMax<<<gridSize, blockSize, smem>>>(devPoints, numInputs, min, max, first);
		first = false;

		numInputs = gridSize.x; //Subsequent iterations (first = false) look at the data in min and max
	}
	while(gridSize.x > 1);
}

void buildOctree(float3* devData, int numdata)
{
	beginTree<<<1,1>>>(devNodeTypes, devIndices, devNodeBounds, devSupernode, minPos, maxPos);
	insert<<<gridSize, blockSize>>>(devNodeTypes, devIndices, devNodeBounds, devData, numdata, devSupernode);
}

void computeCOM(float3* data, float* mass, int numItems)
{
	sumMassNaive<<<gridSize, blockSize>>>(devNodeTypes, devIndices, devNodeBounds, devSupernode, devInfo, data, mass, numItems);
	divideMassNaive<<<dim3((numArr+blockSize.x-1)/blockSize.x), blockSize>>>(devInfo, numArr);	
}

void barnesHut(float3* pos, float3* acc, float3* vel, float* mass, int numItems)
{
	cuBarnesHut<<<gridSize, blockSize>>>(devNodeTypes, devIndices, devNodeBounds, devSupernode, devInfo, pos, acc, mass, numItems);
	integrateEuler<<<gridSize, blockSize>>>(pos, vel, acc, numItems);
}

void resetOctree()
{
	cudaMemset(devNodeTypes, 0, sizeof(int)*numArr);
	cudaMemset(devIndices, 0, sizeof(int)*numArr);
//	cudaMemset(devNodeBounds, 0, sizeof(bounds)*numArr);
	cudaMemset(devInfo, 0, sizeof(bhInfo)*numArr);

	int one = 1;
	
	cudaMemcpyToSymbol(lastins, &one, sizeof(int), 0, cudaMemcpyHostToDevice);
//	cudaMemcpy("lastins", &one, sizeof(int), cudaMemcpyHostToDevice);
}

void physicsInit(int numPoints)
{
	gridSize = dim3((numPoints+blockSize.x-1)/blockSize.x);
	cudaMalloc((void**)&minPos, gridSize.x*sizeof(float3));
	cudaMalloc((void**)&maxPos, gridSize.x*sizeof(float3));

	cudaMalloc((void**)&devNodeTypes, numArr*sizeof(int));
	cudaMalloc((void**)&devIndices, numArr*sizeof(int));
	cudaMalloc((void**)&devNodeBounds, numArr*sizeof(bounds));
	cudaMalloc((void**)&devSupernode, numArr*sizeof(int));
	cudaMalloc((void**)&devInfo, numArr*sizeof(bhInfo));

	resetOctree();
}

#include <iostream>
float hostTheta = -1;

void runPhysics(float3* devPos, float3* devVel, float3* devAcc, float* devMass, int numPoints, bool useBarnesHut, float userTheta)
{
	if(userTheta != hostTheta)
	{
		hostTheta = userTheta;
		cudaMemcpyToSymbol(theta, &userTheta, sizeof(float), 0, cudaMemcpyHostToDevice);
	}

	if(useBarnesHut)
	{
		
		if(!init)
		{
			physicsInit(numPoints);
			init = true;
		}

		get3DMinMax(devPos, numPoints, minPos, maxPos);
		buildOctree(devPos, numPoints);
		computeCOM(devPos, devMass, numPoints);
		barnesHut(devPos, devAcc, devVel, devMass, numPoints);
		resetOctree();
	}
	else
	{
		int smem = sizeof(float4)*blockSize.x;
		allPairsNormal<<<gridSize, blockSize, smem>>>(devPos, devAcc, devMass, numPoints, devVel);
	}







	cudaDeviceSynchronize();
	std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;

	//sumMasses();

	//allPairsNormal<<<gridSize, blockSize, smem>>>(devPos, devAcc, devMass, numPoints, devVel);


	


//	cudaDeviceSynchronize();
//	printf3<<<1,1>>>(minPos);
//	printf3<<<1,1>>>(maxPos);
//	cudaDeviceSynchronize();
//	integrateEuler<<<gridSize, blockSize>>>(devPos, devVel, devAcc, numPoints);
}
