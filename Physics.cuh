#ifndef PHYSICS_CUH
#define PHYSICS_CUH

void runPhysics(float3* devPos, float3* devVel, float3* devAcc, float* devMass, int numPoints, bool useBarnesHut, float theta);
void physicsInit(int numPoints);


#endif