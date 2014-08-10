#version 400

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 vel;
layout(location = 2) in vec3 acc;
layout(location = 3) in float mass;

uniform mat4 proj;
uniform mat4 view;

out vec3 col;

void main()
{
	col = vel;
	gl_Position = proj * view * vec4(pos, 1.0f);
}