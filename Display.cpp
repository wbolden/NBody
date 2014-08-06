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
	numVerts = 0;
	vbo = 0;
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

void Display::setVertexData(GLfloat* points, unsigned int numVerts)
{
	this->numVerts = numVerts;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*numVerts*3, points, GL_DYNAMIC_DRAW);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
}

float3* Display::getCUDAVBOPointer()
{
	float3* cudavbo = 0;

	cudaGraphicsGLRegisterBuffer(&cudavboRes, vbo, cudaGraphicsMapFlagsNone);
	cudaGraphicsMapResources(1, &cudavboRes, 0);

	size_t numBytes = numVerts*sizeof(GLfloat)*3;

	cudaGraphicsResourceGetMappedPointer((void**)&cudavbo, &(numBytes), cudavboRes);
	return cudavbo;
}

void Display::unmapCUDARES()
{
	cudaGraphicsUnmapResources(1, &cudavboRes, 0);
	cudaGraphicsUnregisterResource(cudavboRes);
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

	glDrawArrays(GL_POINTS, 0, numVerts);

	glfwSwapBuffers(window);
	glfwPollEvents();
}

Display::~Display(void)
{
	printf("called\n");
	glfwDestroyWindow(window);
	glfwTerminate();
}
