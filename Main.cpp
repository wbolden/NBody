#include "Display.h"
#include <cstdio>
#include <stdlib.h>
#include <time.h>
#include "Physics.cuh"
#include "Timer.h"

#define WIDTH 1280
#define HEIGHT 720


int main()
{

	Display display = Display(WIDTH, HEIGHT);
	display.initShaders();

	int num = 16384*2;

	

	size_t size = num * sizeof(GLfloat) * 3;

	GLfloat* points = (GLfloat*) malloc(size);
	GLfloat* vels = (GLfloat*) malloc(size);
	GLfloat* accel = (GLfloat*) malloc(size);
	GLfloat* masses = (GLfloat*)malloc(num * sizeof(GLfloat));

	srand(time(0));

	for(int i = 0; i < num; i++)
	{
		int x = i*3;
		int y = i*3 +1;
		int z = i*3 +2;

		points[i*3] = (float)(rand() % 1000) / 100.0f -5.0f;
		points[i*3+1] = (float)(rand() % 1000) / 100.0f -5.0f;
		points[i*3+2] = (float)(rand() % 1000) / 100.0f -5.0f;


	//	points[i*3+1] = 0;

		vels[i*3] = 0;
		vels[i*3+1] = 0;
		vels[i*3+2] = 0;

		if(points[x] < 0)
		{
			if(points[z]< 0)
			{
				vels[x] ++;
			}
			else
			{
				vels[z] --;
			}
		}
		else
		{
			if(points[z]< 0)
			{
				vels[z]++;
			}
			else
			{
				vels[x]--;
			}

		}

		vels[x]/= 10.0f;
		vels[z]/=10.0f;

		masses[i] = 1000000.0f +  rand()%1000000;

		//masses[i] = 10000000.0f + rand()  - rand();

	}





	display.setVertexData(points, vels, accel, masses, num);

	float3* p = 0;
	float3* v = 0;
	float3* a = 0;
	float* m = 0;

	display.registerCUDA();

	Timer timer = Timer(); 

	while(display.running()) 
	{
		timer.start();

		if(!display.paused())
		{
			display.getDevicePointers(&p, &v, &a, &m);
			runPhysics(p, v, a, m, display.getNumPoints());
			display.unmapCUDAResources();
		}
		display.render();

		timer.stop();
		printf("%f\n", timer.getElapsed());
	}

	display.unregisterCUDA();

	return 0;
}