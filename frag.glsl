#version 400

out vec4 fragColor;

in vec3 col;

void main()
{
	float d = max(col.x, max(col.y, col.z));
	if(d < 1.0f)
	{
		d = 1.0f;
	}
	fragColor = vec4(col/d, 1.0);
}