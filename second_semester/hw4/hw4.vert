#version 430 core

varying vec3 View;
varying vec3 Light;
varying vec3 Normal;
varying vec2 TexCoord;

void main()
{
   //  Vertex location in modelview coordinates
   vec4 P = gl_ModelViewMatrix * gl_Vertex;
   //  Light position
   Light  = gl_LightSource[0].position.xyz - P.xyz;
   //  Normal
   Normal = gl_NormalMatrix * gl_Normal;
   //  Eye position
   View  = -P.xyz;
   //  Set vertex position
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
   //  Set texture coordinate
   TexCoord = gl_MultiTexCoord0.xy;
}
