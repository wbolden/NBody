#include "Display.h"
#include <cstdio>

#define WIDTH 800
#define HEIGHT 600


int main()
{

	Display display = Display(WIDTH, HEIGHT);

	while(true)
	{
		display.displayFrame();
	}

	const GLubyte* version = glGetString(GL_RENDERER);

	printf("%s\n", version);


	return 0;
}