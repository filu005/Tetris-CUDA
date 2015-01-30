#version 330 core
layout (location = 0) in vec2 position;

out vec3 outColor;

uniform vec3 color;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * model * vec4(position.x, position.y, 0.0, 1.0);
    outColor = color;
}