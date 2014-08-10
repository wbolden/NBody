#include "Display.h"
#include <cstdio>
#include <stdlib.h>
#include <time.h>
#include "Physics.cuh"

#define WIDTH 800
#define HEIGHT 600


int main()
{

	Display display = Display(WIDTH, HEIGHT);
	display.initShaders();

	int num = 1000;
	size_t size = num * sizeof(GLfloat) * 3;

	GLfloat* points = (GLfloat*) malloc(size);
	GLfloat* vels = (GLfloat*) malloc(size);
	GLfloat* accel = (GLfloat*) malloc(size);
	GLfloat* masses = (GLfloat*)malloc(num * sizeof(GLfloat));

	srand(time(0));

	for(int i = 0; i < num; i++)
	{
		points[i*3] = (float)(rand() % 1000) / 100.0f -5.0f;
		points[i*3+1] = (float)(rand() % 100) / 100.0f -5.0f;
		points[i*3+2] = (float)(rand() % 1000) / 100.0f -5.0f;

		vels[i*3] = 0;
		vels[i*3+1] = 0;
		vels[i*3+2] = 0;

		masses[i] = 100000.0f +  rand()%10000000;

		//masses[i] = 10000000.0f + rand()  - rand();

	}

//	masses[32] = 900000000.0f;



/*
	GLfloat points[] = 
	{
		0.0f, 0.5f, -1.0f,
		0.5f, -0.5f, -1.0f,
		-0.5f, -0.5f, -1.0f
	};

	GLfloat vels[] =
	{
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f
	};

	GLfloat accel[] =
	{
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f
	};


	GLfloat masses[] =
	{	
		1000000.6f, 1000000.5f, 1000000.3f
	};
*/
	display.setVertexData(points, vels, accel, masses, num);

	float3* p = 0;
	float3* v = 0;
	float3* a = 0;
	float* m = 0;

	display.registerCUDA();

	while(display.running()) 
	{
		display.getDevicePointers(&p, &v, &a, &m);
		runPhysics(p, v, a, m, display.getNumPoints());
		display.unmapCUDAResources();

		display.render();
	}

	display.unregisterCUDA();

	return 0;
}