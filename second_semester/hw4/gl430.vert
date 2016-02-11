#version 430 core

//  Transformation matrices
uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;

//  Vertex attributes (input)
layout(location = 0) in vec4 Vertex;
layout(location = 1) in vec3 Color;

//  Output to next shader
out vec3 FrontColor;

void main()
{	
   //  Pass colors to fragment shader (will be interpolated)
   FrontColor = Color;
   //  Set transformed vertex location
   gl_Position =  ProjectionMatrix * ModelViewMatrix * Vertex;
}
