// extremely generic vertex shader

varying vec4 Color;
void main()
{
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
   Color = gl_Position/2.0 + 0.5;
}
