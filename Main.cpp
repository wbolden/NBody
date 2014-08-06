#include "Display.h"
#include <cstdio>
#include "Physics.cuh"

#define WIDTH 800
#define HEIGHT 600


int main()
{

	Display display = Display(WIDTH, HEIGHT);
	display.initShaders();


	GLfloat points[] = 
	{
		0.0f, 0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f, -0.5f, 0.0f
	};

	GLfloat vels[] =
	{

		0.0f, 0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f, -0.5f, 0.0f
	};


	GLfloat masses[] =
	{	
		0.6f, 0.5f, 0.3f
	};

	display.setVertexData(points, vels, masses, 3);

	float3* p = 0;
	float3* v = 0;
	float* m = 0;

	display.registerCUDA();

	while(true)
	{
		display.getCUDAVBOPointers(&p, &v, &m);
		runPhysics(p, v, m);
		display.unmapCUDARES();

		display.displayFrame();
	}

	display.unregisterCUDA();

	return 0;
}