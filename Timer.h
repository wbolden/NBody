#ifndef TIMER_H
#define TIMER_H
class Timer
{
public:
	Timer(void);

	void start(void);
	void stop(void);

	double getElapsed(void);
	long long getFrameCount(void);
	double getFPS(void);

	~Timer(void);

private:
	double elapsedTime;
	double startTime;
	long long frameCount;
};

#endif