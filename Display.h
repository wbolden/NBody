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

	void getCUDAVBOPointers(float3** pos, float3** vel, float** mass);
	void unmapCUDARES();

	void unregisterCUDA();

	void displayFrame();

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
	
	cudaGraphicsResource_t cudaResources[3];
};

#endif
