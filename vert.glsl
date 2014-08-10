#version 400

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 vel;
layout(location = 2) in float mass;

uniform mat4 proj;
uniform mat4 view;


void main()
{
	gl_Position = proj * view * vec4(pos, 1.0f);
}