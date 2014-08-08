#ifndef DISPLAY_H
#define DISPLAY_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glu.h>

class Display
{
public:
	Display(int width, int height);
	void initShaders();
	void setVertexData(GLfloat* points, GLfloat* velocities, GLfloat* masses ,unsigned int numPoints);

	void registerCUDA();

	void getDevicePointers(float3** pos, float3** vel, float** mass);
	void unmapCUDAResources();

	void unregisterCUDA();

	bool running();

	void render();

	~Display(void);

private:
	GLFWwindow* window;
	int width, height;
	
	GLuint vboPos;
	GLuint vboVel;
	GLuint vboMass;

	GLuint vao;
	GLuint shaderProgram;

	int numPoints;

	size_t posBytes;
	size_t velBytes;
	size_t massBytes;

	float* matrix;

	cudaGraphicsResource_t cudaResources[3];
};

#endif
