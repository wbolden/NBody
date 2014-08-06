#include <cstdio>

__global__ void cutest(float3* verts)
{
	int x = threadIdx.x;

	//printf("%f\n%f\n%f\n\n", verts[x].x,  verts[x].y,  verts[x].z);

	verts[x].x -= 0.0001f;

}

void runPhysics(float3* devVerts)
{

	cutest<<<1, 3>>>(devVerts);
}