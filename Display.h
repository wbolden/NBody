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
	void setVertexData(GLfloat* points, unsigned int length);
	float3* getCUDAVBOPointer();
	void unmapCUDARES();
	void displayFrame();

	~Display(void);

private:
	GLFWwindow* window;
	int width, height;
	
	GLuint vbo;
	GLuint vao;
	GLuint shaderProgram;

	int numVerts;

	cudaGraphicsResource_t cudavboRes;
};

#endif
