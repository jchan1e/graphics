#version 410 core

uniform mat4 ModelViewMatrix;
uniform mat4 ProjectionMatrix;
//uniform vec4 LightPos;
layout(location=0) in vec4 Vertex;
//layout(location=1) in vec3 Normal;
//layout(location=2) in vec3 Color;
//layout(location=3) in vec2 Tex;
//
//out vec3 View;
//out vec3 Light;
//out vec3 Norm;
//out vec2 TexCoord;
//out vec4 Front_Color;

void main()
{
   //  Vertex location in modelview coordinates
   //vec4 P = ModelViewMatrix * Vertex;
   ////  Light position
   //Light  = LightPos.xyz - P.xyz;
   ////  Normal
   //NormalMatrix = transpose(inverse(mat3(ModelViewMatrix)))
   //Norm = NormalMatrix * Normal;
   ////  Eye position
   //View  = -P.xyz;
   //  Set vertex position
   gl_Position = ModelViewMatrix * ProjectionMatrix * Vertex;
   //  Set texture coordinate
   //TexCoord = Tex;
   //Front_Color = Color;
}
