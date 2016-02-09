#version 430 core

//  Input from previous shader
in vec3 FrontColor;

//  Fragment color
layout (location=0) out vec4 Fragcolor;

void main()
{
   Fragcolor = vec4(FrontColor,1.0);
}
