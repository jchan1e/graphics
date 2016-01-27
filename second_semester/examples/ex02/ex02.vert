//  NDC RGB vertex shader
void main()
{
   //  Set vertex coordinates
   //  This is equivalent to ftransform()
   gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;

   //  Map 4D to 3D and map to [-1,+1] to [0,1] in all three dimensions
   gl_FrontColor = 0.5*(gl_Position/gl_Position.w + 1.0);
}
