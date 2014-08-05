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

	glfwInit();

	glfwWindowHint(GLFW_SAMPLES, 16);

	window = glfwCreateWindow(width, height, "NBody Simulation", NULL, NULL);
	glfwMakeContextCurrent(window);





	glewExperimental = GL_TRUE;
	glewInit();

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	GLfloat points[] = 
	{
		0.0f, 0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f, -0.5f, 0.0f
	};

	vbo = 0;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);

	vao = 0;

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);





	char* vshader = loadShader("vert.glsl");

//	printf("%s\n", vshader);	

	char* fshader = loadShader("frag.glsl");

//	printf("%s\n", fshader);

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
/*

	glPointSize(3.0f);
	glColor4f(1, 1, 1, 1);

	glBegin(GL_POINTS);

	for(int i = -10; i < 10; i++)
	{
		glVertex3f(i*0.4, i*0.3, i*0.2);
	}

	glEnd();
*/
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(shaderProgram);
	glBindVertexArray(vao);

	glDrawArrays(GL_POINTS, 0, 3);

	glfwSwapBuffers(window);
	glfwPollEvents();
}

Display::~Display(void)
{
	printf("called\n");
	glfwDestroyWindow(window);
	glfwTerminate();
}
