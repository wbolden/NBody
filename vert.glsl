#version 400

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 vel;
layout(location = 2) in vec3 acc;
layout(location = 3) in float mass;

uniform mat4 proj;
uniform mat4 view;
uniform int mode;

out vec3 col;

void main()
{
	if(mode == 0)
	{
		col.x = 1.0f;
		col.y = 1.3f;
		col.z = 1.3f;
	}
	else if(mode == 1)
	{
		col = abs(acc)*10*10;
	}
	else
	{
		col = abs(vel)*10;
	}

	gl_Position = proj * view * vec4(pos, 1.0f);

	vec3 ndc = gl_Position.xyz/gl_Position.w;

	gl_PointSize = (1.005f- ndc.z) * 100;
	//gl_PointSize = 30;
}