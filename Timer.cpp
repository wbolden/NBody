#include "Timer.h"
#include <GLFW/glfw3.h>

Timer::Timer(void)
{
	startTime = glfwGetTime();
	elapsedTime = glfwGetTime() - startTime;
	frameCount = 0;
}

void Timer::start(void)
{
	startTime = glfwGetTime();
}

void Timer::stop(void)
{
	elapsedTime = glfwGetTime() - startTime;
	frameCount++;
}

double Timer::getElapsed(void)
{
	return elapsedTime;
}


long long Timer::getFrameCount(void)
{
	return frameCount;
}

double Timer::getFPS(void)
{
	return 1/elapsedTime;
}


Timer::~Timer(void)
{
}
