#include "Display.h"
#include <cuda_gl_interop.h>
#include <fstream>

#define ONED 0.01745329f

bool Display::pressed(int key)
{
	return glfwGetKey(window, key);
}

void Display::moveForeward(float ammount)
{
	float rotxr = rot.x;
	float rotyr = rot.y;

	pos.x += -ammount*sinf(rotxr)*cosf(rotyr);
	pos.y += -ammount*sinf(rotyr);
	pos.z += ammount*cosf(rotxr)*cosf(rotyr);
}


void Display::moveRight(float ammount)
{
	float rotxr = rot.x;

	pos.x += ammount*cosf(rotxr);
	pos.z += ammount*sinf(rotxr);
}


void Display::handleInput()
{
	float moveSpeed = 0.05f;
	float rotSpeed = 0.35f;

	if(pressed(GLFW_MOD_SHIFT))
	{
		moveSpeed *= 10;
	}

	if(pressed(GLFW_KEY_ESCAPE))
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}

	if(pressed(GLFW_KEY_W))
	{
		moveForeward(moveSpeed);
	}
	if(pressed(GLFW_KEY_S))
	{
		moveForeward(-moveSpeed);
	}
	if(pressed(GLFW_KEY_A))
	{
		moveRight(moveSpeed);
	}
	if(pressed(GLFW_KEY_D))
	{
		moveRight(-moveSpeed);
	}


	if(pressed(GLFW_KEY_UP))
	{
		rot.y += rotSpeed*ONED;
	}
	if(pressed(GLFW_KEY_DOWN))
	{
		rot.y -= rotSpeed*ONED;
	}
	if(pressed(GLFW_KEY_RIGHT))
	{
		rot.x += rotSpeed*ONED;
	}
	if(pressed(GLFW_KEY_LEFT))
	{
		rot.x -= rotSpeed*ONED;
	}

	if(pressed(GLFW_KEY_V))
	{
		color = 2;
	}
	if(pressed(GLFW_KEY_C))
	{
		color = 1;
	}
	if(pressed(GLFW_KEY_N))
	{
		color = 0;
	}


}

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
	color = 0;

	pos = glm::vec3(0.0f, 0.0f, -2.0f);
	rot = glm::vec3(0.0f, 0.0f, 0.0f);

	glfwInit();

	glfwWindowHint(GLFW_SAMPLES, 16);

	window = glfwCreateWindow(width, height, "NBody Simulation", NULL, NULL);
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	glewInit();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

}

void Display::setVertexData(GLfloat* points, GLfloat* velocities, GLfloat* accelerations, GLfloat* masses ,unsigned int numPoints)
{
	this->numPoints = numPoints;

	posBytes = sizeof(GLfloat)*numPoints*3;
	velBytes = sizeof(GLfloat)*numPoints*3;
	accBytes = sizeof(GLfloat)*numPoints*3;
	massBytes = sizeof(GLfloat)*numPoints*1;

	glGenBuffers(1, &vboPos);
	glBindBuffer(GL_ARRAY_BUFFER, vboPos);
	glBufferData(GL_ARRAY_BUFFER, posBytes, points, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vboVel);
	glBindBuffer(GL_ARRAY_BUFFER, vboVel);
	glBufferData(GL_ARRAY_BUFFER, velBytes, velocities, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vboAcc);
	glBindBuffer(GL_ARRAY_BUFFER, vboAcc);
	glBufferData(GL_ARRAY_BUFFER, accBytes, accelerations, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vboMass);
	glBindBuffer(GL_ARRAY_BUFFER, vboMass);
	glBufferData(GL_ARRAY_BUFFER, massBytes, masses, GL_DYNAMIC_DRAW);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vboPos);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, vboVel);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, vboAcc);
	glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glEnableVertexAttribArray(3);
	glBindBuffer(GL_ARRAY_BUFFER, vboMass);
	glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 0, NULL);
}

void Display::registerCUDA()
{
	cudaGraphicsGLRegisterBuffer(&cudaResources[0], vboPos, cudaGraphicsMapFlagsNone);
	cudaGraphicsGLRegisterBuffer(&cudaResources[1], vboVel, cudaGraphicsMapFlagsNone);
	cudaGraphicsGLRegisterBuffer(&cudaResources[2], vboAcc, cudaGraphicsMapFlagsNone);
	cudaGraphicsGLRegisterBuffer(&cudaResources[3], vboMass, cudaGraphicsMapFlagsNone);	
}

void Display::getDevicePointers(float3** pos, float3** vel, float3** acc, float** mass)
{
	cudaGraphicsMapResources(4, cudaResources, 0);

	cudaGraphicsResourceGetMappedPointer((void**)pos, &(posBytes), cudaResources[0]);
	cudaGraphicsResourceGetMappedPointer((void**)vel, &(velBytes), cudaResources[1]);
	cudaGraphicsResourceGetMappedPointer((void**)acc, &(accBytes), cudaResources[2]);
	cudaGraphicsResourceGetMappedPointer((void**)mass, &(massBytes), cudaResources[3]);
}


int Display::getNumPoints()
{
	return numPoints;
}

void Display::unmapCUDAResources()
{
	cudaGraphicsUnmapResources(4, cudaResources, 0);
}

void Display::unregisterCUDA()
{
	cudaGraphicsUnregisterResource(cudaResources[0]);
	cudaGraphicsUnregisterResource(cudaResources[1]);
	cudaGraphicsUnregisterResource(cudaResources[2]);
	cudaGraphicsUnregisterResource(cudaResources[3]);
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


void Display::render()
{
	handleInput();

	glm::mat4 proj = glm::perspective(70.0f, (float)width/(float)height, 0.1f, 1000.0f);
	glm::mat4 view = glm::mat4(1.0f);
	
	view = glm::rotate(view, rot.y, glm::vec3(-1.0f, 0.0f, 0.0f));
	view = glm::rotate(view, rot.x, glm::vec3(0.0f, 1.0f, 0.0f));
	view = glm::translate(view, pos);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(shaderProgram);

	int projMatrixLocation = glGetUniformLocation(shaderProgram, "proj");
	int viewMatrixLocation = glGetUniformLocation(shaderProgram, "view");
	glUniformMatrix4fv(projMatrixLocation, 1, GL_FALSE, glm::value_ptr(proj));
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, glm::value_ptr(view));

	int colorMode = glGetUniformLocation(shaderProgram, "mode");
	glUniform1i(colorMode, color);

	glBindVertexArray(vao);

	glDrawArrays(GL_POINTS, 0, numPoints);

	glfwSwapBuffers(window);
	glfwPollEvents();
}

bool Display::running()
{
	return !glfwWindowShouldClose(window);
}

Display::~Display(void)
{
	glfwDestroyWindow(window);
	glfwTerminate();
}
