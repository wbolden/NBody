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

	display.setVertexData(points, 3);


	while(true)
	{

		runPhysics(display.getCUDAVBOPointer());
		display.unmapCUDARES();

		display.displayFrame();
	}

	return 0;
}