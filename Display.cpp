#include "Display.h"
#include <cuda_gl_interop.h>
#include <fstream>

char* loadShader(const char* filename)
{
	std::ifstream file;
	unsigned int length;
	char* shader = NULL;

	file.open(filename);

	file.seekg(0, std::ios::end);
	length = file.tellg();
	file.clear();
	file.seekg(0, std::ios::beg);

	shader = new char[length + 1];
	file.read(shader, sizeof(char)*length);
	shader[length] = 0;

	file.close();
	return shader;
}


Display::Display(int width, int height)
{
	this->width = width;
	this->height = height;
	numPoints = 0;
	vao = 0;

	glfwInit();

	glfwWindowHint(GLFW_SAMPLES, 16);

	window = glfwCreateWindow(width, height, "NBody Simulation", NULL, NULL);
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	glewInit();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
}

void Display::setVertexData(GLfloat* points, GLfloat* velocities, GLfloat* masses ,unsigned int numPoints)
{
	this->numPoints = numPoints;

	posBytes = sizeof(GLfloat)*numPoints*3;
	velBytes = sizeof(GLfloat)*numPoints*3;
	massBytes = sizeof(GLfloat)*numPoints*1;

	glGenBuffers(1, &vboPos);
	glBindBuffer(GL_ARRAY_BUFFER, vboPos);
	glBufferData(GL_ARRAY_BUFFER, posBytes, points, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vboVel);
	glBindBuffer(GL_ARRAY_BUFFER, vboVel);
	glBufferData(GL_ARRAY_BUFFER, velBytes, velocities, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vboMass);
	glBindBuffer(GL_ARRAY_BUFFER, vboMass);
	glBufferData(GL_ARRAY_BUFFER, massBytes, masses, GL_DYNAMIC_DRAW);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vboPos);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
}

void Display::getCUDAVBOPointers(float3** pos, float3** vel, float** mass)
{
	cudaGraphicsGLRegisterBuffer(&cudavboPosRes, vboPos, cudaGraphicsMapFlagsNone);
	cudaGraphicsMapResources(1, &cudavboPosRes, 0);

	cudaGraphicsGLRegisterBuffer(&cudavboVelRes, vboVel, cudaGraphicsMapFlagsNone);
	cudaGraphicsMapResources(1, &cudavboVelRes, 0);

	cudaGraphicsGLRegisterBuffer(&cudavboMassRes, vboMass, cudaGraphicsMapFlagsNone);
	cudaGraphicsMapResources(1, &cudavboMassRes, 0);

	cudaGraphicsResourceGetMappedPointer((void**)pos, &(posBytes), cudavboPosRes);
	cudaGraphicsResourceGetMappedPointer((void**)vel, &(velBytes), cudavboVelRes);
	cudaGraphicsResourceGetMappedPointer((void**)mass, &(massBytes), cudavboMassRes);
}

void Display::unmapCUDARES()
{
	cudaGraphicsUnmapResources(1, &cudavboPosRes, 0);
	cudaGraphicsUnregisterResource(cudavboPosRes);

	cudaGraphicsUnmapResources(1, &cudavboVelRes, 0);
	cudaGraphicsUnregisterResource(cudavboVelRes);

	cudaGraphicsUnmapResources(1, &cudavboMassRes, 0);
	cudaGraphicsUnregisterResource(cudavboMassRes);
}

void Display::initShaders()
{
	char* vshader = loadShader("vert.glsl");
	char* fshader = loadShader("frag.glsl");

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, (const GLchar**)&vshader, NULL);
	glCompileShader(vs);

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, (const GLchar**)&fshader, NULL);
	glCompileShader(fs);

	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, fs);
	glAttachShader(shaderProgram, vs);
	glLinkProgram(shaderProgram);
}

void Display::displayFrame()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(shaderProgram);
	glBindVertexArray(vao);

	glDrawArrays(GL_POINTS, 0, numPoints);

	glfwSwapBuffers(window);
	glfwPollEvents();
}

Display::~Display(void)
{
	printf("called\n");
	glfwDestroyWindow(window);
	glfwTerminate();
}
