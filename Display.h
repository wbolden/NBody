#ifndef DISPLAY_H
#define DISPLAY_H

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <GL/glu.h>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Display
{
public:
	Display(int width, int height, bool fullscreen);
	void initShaders();
	void setVertexData(GLfloat* points, GLfloat* velocities, GLfloat* accelerations, GLfloat* masses ,unsigned int numPoints);

	void registerCUDA();

	void getDevicePointers(float3** pos, float3** vel, float3** acc, float** mass);
	int getNumPoints();
	void unmapCUDAResources();

	void unregisterCUDA();

	bool running();
	bool paused();
	void render();

	~Display(void);

private:
	void handleInput();
	bool pressed(int key);
	void moveForeward(float ammount);
	void moveRight(float ammount);

	GLFWwindow* window;
	int width, height;
	
	GLuint vboPos;
	GLuint vboVel;
	GLuint vboAcc;
	GLuint vboMass;

	GLuint vao;
	GLuint shaderProgram;

	int numPoints;

	size_t posBytes;
	size_t velBytes;
	size_t accBytes;
	size_t massBytes;

	cudaGraphicsResource_t cudaResources[4];

	glm::vec3 pos;
	glm::vec3 rot;

	GLint color;
	bool pause;
	bool clear;

};


#endif
