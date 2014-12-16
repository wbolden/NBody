#include "Display.h"
#include <cuda_gl_interop.h>
#include <fstream>

#define ONED 0.01745329f

const char* defaultVShader = "#version 400 \n"
"layout(location = 0) in vec3 pos;"
"layout(location = 1) in vec3 vel;"
"layout(location = 2) in vec3 acc;"
"layout(location = 3) in float mass;"
"uniform mat4 proj;"
"uniform mat4 view;"
"uniform int mode;"
"out vec3 col;"
"void main()"
"{"
"	if(mode == 0)"
"	{"
"		col.x = 1.0f;"
"		col.y = 1.3f;"
"		col.z = 1.3f;"
"	}"
"	else if(mode == 1)"
"	{"
"		col = abs(acc)*10*10;"
"	}"
"	else"
"	{"
"		col = abs(vel)*10;"
"	}"
"	gl_Position = proj * view * vec4(pos, 1.0f);"
"	vec3 ndc = gl_Position.xyz/gl_Position.w;"
"	gl_PointSize = (1.005f- ndc.z) * 100;"
"}";

const char* defaultFShader = "#version 400 \n"
"out vec4 fragColor;"
"in vec3 col;"
"void main()"
"{"
"	float d = max(col.x, max(col.y, col.z));"
"	if(d < 1.0f)"
"	{"
"		d = 1.0f;"
"	}"
"	fragColor = vec4(col/d, 1.0);"
//"	fragColor = vec4(col, 1.0);"
"}";


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
	clear = false;

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
	pause = false;

	if(pressed(GLFW_KEY_SPACE))
	{
		pause = true;
	}

	clear = false;
	if(pressed(GLFW_KEY_Y))
	{
		clear = true;
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


Display::Display(int width, int height, bool fullscreen)
{
	this->width = width;
	this->height = height;
	numPoints = 0;
	vao = 0;
	color = 0;
	pause = false;
	clear = false;

	pos = glm::vec3(0.0f, 1.0f, -5.0f);
	rot = glm::vec3(0.0f, 0.0f, 0.0f);

	glfwInit();

	glfwWindowHint(GLFW_SAMPLES, 16);
	if(fullscreen)
	{
		window = glfwCreateWindow(width, height, "NBody Simulation", glfwGetPrimaryMonitor(), NULL);
	}
	else
	{
		window = glfwCreateWindow(width, height, "NBody Simulation", NULL, NULL);
	}

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	glewInit();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

}

bool Display::paused()
{
	return pause;
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

	if(clear)
	{
		cudaMemset(*vel, 0, velBytes);
	}
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

bool shaderCompiled(GLuint shader)
{
	GLint status;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
	return (bool)status;
}

bool compileShader(GLuint shader, const GLchar* shaderSource)
{
	glShaderSource(shader, 1, &shaderSource, NULL);
	glCompileShader(shader);

	return shaderCompiled(shader);
}

void Display::initShaders()
{
	const char* vshader = loadShader("vert.glsl");
	const char* fshader = loadShader("frag.glsl");

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);

	if(!compileShader(vs, (const GLchar*)vshader))
	{
		vshader = defaultVShader;
		compileShader(vs, (const GLchar*)vshader);
	}
	if(!compileShader(fs, (const GLchar*)fshader))
	{
		fshader = defaultFShader;
		compileShader(fs, (const GLchar*)fshader);
	}

	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, fs);
	glAttachShader(shaderProgram, vs);
	glLinkProgram(shaderProgram);
}


void Display::render()
{
	handleInput();

	glm::mat4 proj = glm::perspective(70.0f, (float)width/(float)height, 0.1f, 1000000.0f);
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
